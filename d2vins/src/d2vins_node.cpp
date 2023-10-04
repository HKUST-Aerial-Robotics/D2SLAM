#include <d2frontend/d2frontend.h>
#include <d2common/d2frontend_types.h>
#include <d2frontend/loop_net.h>
#include <d2frontend/loop_detector.h>
#include "sensor_msgs/Imu.h"
#include "estimator/d2estimator.hpp"
#include "network/d2vins_net.hpp"
#include <mutex>
#include <queue>
#include <chrono>
#include <d2frontend/d2featuretracker.h>
#include <swarm_msgs/swarm_fused.h>
#include <spdlog/spdlog.h>

using namespace std::chrono;
#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}

using namespace D2VINS;
using namespace D2Common;
using namespace std::chrono;

class D2VINSNode :  public D2FrontEnd::D2Frontend
{
    typedef std::lock_guard<std::mutex> Guard;
    D2Estimator * estimator = nullptr;
    D2VINSNet * d2vins_net = nullptr;
    ros::Subscriber imu_sub, pgo_fused_sub;
    ros::Publisher visual_array_pub;
    int frame_count = 0;
    std::queue<D2Common::VisualImageDescArray> viokf_queue;
    std::mutex queue_lock;
    ros::Timer estimator_timer, solver_timer;
    std::thread thread_comm, thread_solver, thread_viokf;
    bool has_received_imu = false;
    double last_imu_ts = 0;
    bool need_solve = false;
    std::set<int> ready_drones;
    std::map<int, Swarm::Pose> pgo_poses;
    std::map<int, std::pair<int, Swarm::Pose>> vins_poses;
protected:
    Swarm::Pose getMotionPredict(double stamp) const override {
        return estimator->getMotionPredict(stamp).first.pose();
    }

    virtual void backendFrameCallback(const D2Common::VisualImageDescArray & viokf) override {
        if (params->estimation_mode < D2Common::SERVER_MODE) {
            Guard guard(queue_lock);
            viokf_queue.emplace(viokf);
        }
        frame_count ++;
    };

    void processRemoteImage(VisualImageDescArray & frame_desc, bool succ_track) override {
        spdlog::debug("[D2VINS] processRemoteImage: frame_id {}, drone_id {}, matched_frame {}, is_keyframe {}, succ_track {}, is_lazy_frame {}, sld_win_status size {}",
                frame_desc.frame_id, frame_desc.drone_id, frame_desc.matched_frame, frame_desc.is_keyframe, succ_track, frame_desc.is_lazy_frame, frame_desc.sld_win_status.size());
        ready_drones.insert(frame_desc.drone_id);
        vins_poses[frame_desc.drone_id] = std::make_pair(frame_desc.reference_frame_id, frame_desc.pose_drone);
        if (params->estimation_mode != D2Common::SINGLE_DRONE_MODE && succ_track &&
                !frame_desc.is_lazy_frame && frame_desc.matched_frame < 0) {
            estimator->inputRemoteImage(frame_desc);
        } else {
            if (params->estimation_mode != D2Common::SINGLE_DRONE_MODE && 
                    frame_desc.sld_win_status.size() > 0) {
                estimator->updateSldwin(frame_desc.drone_id, frame_desc.sld_win_status);
            }
            if (frame_desc.matched_frame < 0) {
                VINSFrame frame(frame_desc);
                estimator->getVisualizer().pubFrame(&frame);
            }
        }
        D2Frontend::processRemoteImage(frame_desc, succ_track);
        if (params->pub_visual_frame) {
            visual_array_pub.publish(frame_desc.toROS());
        }
    }

    void updateOutModuleSldWinAndLandmarkDB() {
        //Frame related operations. Need to be protected by frame_mutex
        const std::lock_guard<std::recursive_mutex> lock(estimator->frame_mutex);
        auto sld_win = estimator->getSelfSldWin();
        auto landmark_db = estimator->getLandmarkDB();
        if (params->enable_loop) {
            loop_detector->updatebyLandmarkDB(estimator->getLandmarkDB());
            loop_detector->updatebySldWin(sld_win);
        }
        feature_tracker->updatebySldWin(sld_win);
        feature_tracker->updatebyLandmarkDB(estimator->getLandmarkDB());
    }

    void processVIOKFThread() {
        while(ros::ok()) {
            if (!viokf_queue.empty()) {
                Utility::TicToc estimator_timer;
                if (viokf_queue.size() > params->warn_pending_frames) {
                    ROS_WARN("[D2VINS] Low efficient on D2VINS::estimator pending frames: %d", viokf_queue.size());
                }
                D2Common::VisualImageDescArray viokf;
                {
                    Guard guard(queue_lock);
                    viokf = viokf_queue.front();
                    viokf_queue.pop();
                }
                bool ret;
                {
                    Utility::TicToc input;
                    ret = estimator->inputImage(viokf);
                    double input_time = input.toc();
                    Utility::TicToc loop;
                    if (viokf.is_keyframe) {
                        addToLoopQueue(viokf);
                    }
                    updateOutModuleSldWinAndLandmarkDB();
                    need_solve = true;
                    if (params->verbose || params->enable_perf_output)
                        printf("[D2VINS] input_time %.1fms, loop detector related takes %.1f ms\n", input_time, loop.toc());
                }

                if (params->pub_visual_frame) {
                    visual_array_pub.publish(viokf.toROS());
                }
                if (params->verbose || params->enable_perf_output)
                    printf("[D2VINS] estimator_timer_callback takes %.1f ms\n", estimator_timer.toc());
                bool discover_mode = false;
                for (auto & id : ready_drones) {
                    if (pgo_poses.find(id) == pgo_poses.end() && !estimator->getState().hasDrone(id)) {
                        discover_mode = true;
                        break;
                    }
                }
                if (ret && D2FrontEnd::params->enable_network) { //Only send keyframes
                    if (params->lazy_broadcast_keyframe && !viokf.is_keyframe && !discover_mode) {
                        continue;
                    }
                    std::set<int> nearbydrones = estimator->getNearbyDronesbyPGOData(vins_poses); 
                    bool force_landmarks = false || discover_mode;
                    if (nearbydrones.size() > 0) {
                        force_landmarks = true;
                        if (params->verbose) {
                            printf("[D2VINS] Nearby drones: ");
                            for (auto & id : nearbydrones) {
                                printf("%d ", id);
                            }
                            printf("\n");
                        }
                    }
                    Utility::TicToc broadcast_timer;
                    loop_net->broadcastVisualImageDescArray(viokf, force_landmarks);
                    if (params->verbose || params->enable_perf_output) {
                        printf("[D2VINS] broadcastVisualImageDescArray takes %.1f ms\n", broadcast_timer.toc());
                    }
                }
            } else {
                usleep(1000);
            }
        }
    }

    virtual void imuCallback(const sensor_msgs::Imu & imu) {
        IMUData data(imu);
        if(!has_received_imu) {
            has_received_imu = true;
            last_imu_ts = imu.header.stamp.toSec();
        }
        data.dt = imu.header.stamp.toSec() - last_imu_ts;
        last_imu_ts = imu.header.stamp.toSec();
        estimator->inputImu(data);
    }

    void pgoSwarmFusedCallback(const swarm_msgs::swarm_fused & fused) {
        if (params->estimation_mode == D2Common::SINGLE_DRONE_MODE) {
            return;
        }
        for (size_t i = 0; i < fused.ids.size(); i++) {
            Swarm::Pose pose(fused.local_drone_position[i], fused.local_drone_rotation[i]);
            pgo_poses[fused.ids[i]] = pose;
            if (params->verbose)
                printf("[D2VINS] PGO fused drone %d: %s\n", fused.ids[i], pose.toStr().c_str());
        }
        estimator->setPGOPoses(pgo_poses);
    }
    
    void distriburedTimerCallback(const ros::TimerEvent & e) {
        estimator->solveinDistributedMode();
        updateOutModuleSldWinAndLandmarkDB();
    }

    void solverThread() {
        while (ros::ok()) {
            if (need_solve) {
                estimator->solveinDistributedMode();
                updateOutModuleSldWinAndLandmarkDB();
                if (!params->consensus_sync_to_start) {
                    need_solve = false;
                } else {
                    usleep(0.5/params->estimator_timer_freq*1e6);
                }
            } else {
                usleep(1000);
            }
        }
    }


    void Init(ros::NodeHandle & nh) {
        D2Frontend::Init(nh);
        initParams(nh);
        estimator = new D2Estimator(params->self_id);
        d2vins_net = new D2VINSNet(estimator, params->lcm_uri);
        estimator->init(nh, d2vins_net);
        visual_array_pub = nh.advertise<swarm_msgs::ImageArrayDescriptor>("image_array_desc", 1);
        imu_sub = nh.subscribe(params->imu_topic, 1000, &D2VINSNode::imuCallback, this, ros::TransportHints().tcpNoDelay()); //We need a big queue for IMU.
        pgo_fused_sub = nh.subscribe("/d2pgo/swarm_fused", 1, &D2VINSNode::pgoSwarmFusedCallback, this, ros::TransportHints().tcpNoDelay());
        thread_viokf = std::thread([&] {
            processVIOKFThread();
            printf("[D2VINS] processVIOKFThread exit.\n");
        });
        if (params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS && params->consensus_sync_to_start) {
            solver_timer = nh.createTimer(ros::Duration(1.0/params->estimator_timer_freq), &D2VINSNode::distriburedTimerCallback, this);
        } else if (params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS) {
            thread_solver = std::thread([&] {
                solverThread();
            });
        }
        thread_comm = std::thread([&] {
            ROS_INFO("Starting d2vins_net lcm.");
            while(0 == d2vins_net->lcmHandle()) {
            }
        });
        ROS_INFO("D2VINS node %d initialized. Ready to start.", params->self_id);
    }

public:
    D2VINSNode(ros::NodeHandle & nh) {
        Init(nh);
    }
};

int main(int argc, char **argv)
{
    spdlog::set_pattern("[%H:%M:%S][%^%l%$]%g:%# %v");
    cv::setNumThreads(1);
    ros::init(argc, argv, "d2vins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    D2VINSNode d2vins(n);
    ros::AsyncSpinner spinner(4);
    spinner.start();
    ros::waitForShutdown();
    return 0;
}

