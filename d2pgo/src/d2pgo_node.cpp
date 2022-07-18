#include <ros/ros.h>
#include "d2pgo.h"
#include "swarm_msgs/ImageArrayDescriptor.h"

namespace D2PGO {
class D2PGONode {
    D2PGO * pgo = nullptr;
    ros::Subscriber frame_sub, remote_frame_sub, loop_sub;
    ros::Timer solver_timer;
    double solver_timer_freq = 10;
    D2PGOConfig config;
    std::map<int, ros::Publisher> path_pubs;
    ros::Publisher path_pub;
    ros::Publisher dpgo_data_pub;
    ros::Publisher drone_traj_pub;
    bool write_to_file = false;
    int write_to_file_step = 5;
    int pub_count = 0;
    std::string output_folder;
    ros::NodeHandle * _nh;
protected:
    void processImageArray(const swarm_msgs::ImageArrayDescriptor & frame_desc) {
        if (frame_desc.is_keyframe) {
            ROS_INFO("[D2PGONode@%d] processKeyImageArray %ld", config.self_id, frame_desc.frame_id);
            auto img_array = VisualImageDescArray(frame_desc);
            pgo->addFrame(VINSFrame(img_array));
        }
    }
    
    void processLoop(const swarm_msgs::LoopEdge & loop_info) {
        ROS_INFO("[D2PGONode@%d] processLoop from %ld to %ld", config.self_id, loop_info.keyframe_id_a, loop_info.keyframe_id_b);
        pgo->addLoop(Swarm::LoopEdge(loop_info));
    }

    void pubTrajs(std::map<int, Swarm::DroneTrajectory> & trajs) {
        for (auto it : trajs) {
            auto drone_id = it.first;
            auto traj = it.second;
            if (path_pubs.find(drone_id) == path_pubs.end()) {
                path_pubs[drone_id] = _nh->advertise<nav_msgs::Path>("pgo_path_" + std::to_string(drone_id), 1000);
            }
            if (drone_id == config.self_id) {
                path_pub.publish(traj.get_ros_path());
            }
            drone_traj_pub.publish(traj.toRos());
            path_pubs[drone_id].publish(traj.get_ros_path());
            if (write_to_file && pub_count % write_to_file_step == 0) {
                std::ofstream csv(output_folder + "/pgo_" + std::to_string(drone_id) + ".csv", std::ios::out);
                for (size_t i = 0; i < traj.trajectory_size(); i++) {
                    double stamp = traj.stamp_by_index(i);
                    auto pose = traj.pose_by_index(i);
                    csv << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << stamp << " " << pose.toStr(true) << std::endl;
                }
                csv.close();
            }
        }
        pub_count++;
    }

    void solverTimerCallback(const ros::TimerEvent & event) {
        if (pgo->solve()) { 
            auto trajs = pgo->getOptimizedTrajs();
            pubTrajs(trajs);
        }
    }

    void Init(ros::NodeHandle & nh) {
        InitParams(nh);
        _nh = &nh;
        pgo = new D2PGO(config);
        path_pub = _nh->advertise<nav_msgs::Path>("pgo_path", 1000);
        drone_traj_pub = _nh->advertise<swarm_msgs::DroneTraj>("pgo_traj", 1000);
        dpgo_data_pub = _nh->advertise<swarm_msgs::DPGOData>("pgo_data", 1000);
        pgo->bd_data_callback = [&] (const DPGOData & data) {
            dpgo_data_pub.publish(data.toROS());
        };
        frame_sub = nh.subscribe("image_array_desc", 1, &D2PGONode::processImageArray, this, ros::TransportHints().tcpNoDelay());
        remote_frame_sub = nh.subscribe("remote_frame_desc", 1, &D2PGONode::processImageArray, this, ros::TransportHints().tcpNoDelay());
        loop_sub = nh.subscribe("loop", 1, &D2PGONode::processLoop, this, ros::TransportHints().tcpNoDelay());
        solver_timer = nh.createTimer(ros::Duration(1.0/solver_timer_freq), &D2PGONode::solverTimerCallback, this);
        printf("[D2PGONode@%d] Initialized\n", config.self_id);
    }

    void InitParams(ros::NodeHandle & nh) {
        std::string vins_config_path;
        nh.param<std::string>("vins_config_path", vins_config_path, "");
        cv::FileStorage fsSettings;
        try {
            fsSettings.open(vins_config_path.c_str(), cv::FileStorage::READ);
            std::cout << "PGO Loaded VINS config from " << vins_config_path << std::endl;
        } catch(cv::Exception ex) {
            std::cerr << "ERROR:" << ex.what() << " Can't open config file" << std::endl;
            exit(-1);
        }
        fsSettings["output_path"] >> output_folder;
        config.mode = static_cast<PGO_MODE>((int) fsSettings["pgo_mode"]);
        nh.param<int>("self_id", config.self_id, -1);
        config.ceres_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;// ceres::DENSE_SCHUR;
        config.ceres_options.num_threads = 1;
        config.ceres_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;// ceres::DOGLEG;
        config.ceres_options.max_solver_time_in_seconds =  fsSettings["pgo_solver_time"];
        config.main_id = 1;
    }
public:
    D2PGONode(ros::NodeHandle & nh) {
        Init(nh);
    }
};
}

int main(int argc, char **argv)
{
    cv::setNumThreads(1);
    ros::init(argc, argv, "d2pgo");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    D2PGO::D2PGONode d2pgonode(n);
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    return 0;
}