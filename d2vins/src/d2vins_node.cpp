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

using namespace std::chrono;
// #define BACKWARD_HAS_DW 1
// #include <backward.hpp>
// namespace backward
// {
//     backward::SignalHandling sh;
// }

using namespace D2VINS;
using namespace D2Common;
using namespace std::chrono;

class D2VINSNode :  public D2FrontEnd::D2Frontend
{
    typedef std::lock_guard<std::mutex> Guard;
    D2Estimator * estimator = nullptr;
    D2VINSNet * d2vins_net = nullptr;
    ros::Subscriber imu_sub;
    int frame_count = 0;
    std::queue<D2Common::VisualImageDescArray> viokf_queue;
    std::mutex queue_lock;
    std::mutex esti_lock;
    ros::Timer estimator_timer, solver_timer;
    std::thread th;
    std::thread th_timer;
protected:
    virtual void frameCallback(const D2Common::VisualImageDescArray & viokf) override {
        if (params->estimation_mode < D2VINSConfig::SERVER_MODE && frame_count % params->frame_step == 0) {
            Guard guard(queue_lock);
            viokf_queue.emplace(viokf);
        }
        frame_count ++;
    };

    void processRemoteImage(VisualImageDescArray & frame_desc) override {
        {
            if (params->estimation_mode != D2VINSConfig::SINGLE_DRONE_MODE) {
                Guard guard(esti_lock);
                estimator->inputRemoteImage(frame_desc);
            }
        }
        if (D2FrontEnd::params->enable_loop) {
            loop_detector->processImageArray(frame_desc);
        }
    }
    
    void distriburedTimerCallback(const ros::TimerEvent & event) {
        Guard guard(esti_lock);
        estimator->solveinDistributedMode();
    }

    void timerCallback(const ros::TimerEvent & event) {
        if (!viokf_queue.empty()) {
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
                Guard guard(esti_lock);
                ret = estimator->inputImage(viokf);
            }
            if (ret && D2FrontEnd::params->enable_network) {
                loop_net->broadcastVisualImageDescArray(viokf);
            }
        }
    }

    virtual void imuCallback(const sensor_msgs::Imu & imu) {
        IMUData data(imu);
        data.dt = 1.0/params->IMU_FREQ; //TODO
        estimator->inputImu(data);
    }

    void myHighResolutionTimerThread() {
        double freq = params->IMAGE_FREQ/params->frame_step;
        int duration_us = floor(1000000.0/freq);
        int allow_err = params->consensus_trigger_time_err_us;
        while (!ros::isShuttingDown()) {
            int64_t usec = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
            int err = std::min(duration_us - usec%duration_us, usec%duration_us);
            if (err < allow_err) {
                // printf("call err %ld us\n", err);
                Guard guard(esti_lock);
                estimator->solveinDistributedMode();
            } else {
                if (duration_us - usec%duration_us > 2*allow_err) {
                    usleep(duration_us - usec%duration_us - 2*allow_err);
                }
                usleep(10);
            }
        }
    }

public:
    D2VINSNode(ros::NodeHandle & nh) {
        initParams(nh);
        Init(nh);
        estimator = new D2Estimator(params->self_id);
        d2vins_net = new D2VINSNet(estimator, params->lcm_uri);
        estimator->init(nh, d2vins_net);
        imu_sub  = nh.subscribe(params->imu_topic, 1, &D2VINSNode::imuCallback, this, ros::TransportHints().tcpNoDelay());
        estimator_timer = nh.createTimer(ros::Duration(1.0/params->estimator_timer_freq), &D2VINSNode::timerCallback, this);
        solver_timer = nh.createTimer(ros::Duration(1.0/params->estimator_timer_freq), &D2VINSNode::distriburedTimerCallback, this);
        // if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS) {
        //     th_timer = std::thread([&] {
        //         myHighResolutionTimerThread();
        //     });
        // }
        th = std::thread([&] {
            ROS_INFO("Starting d2vins_net lcm.");
            while(0 == d2vins_net->lcmHandle()) {
            }
        });
        ROS_INFO("D2VINS node %d initialized. Ready to start.", params->self_id);
    }
};

int main(int argc, char **argv)
{
    cv::setNumThreads(1);
    ros::init(argc, argv, "d2vins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    D2VINSNode d2vins(n);
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    return 0;
}

