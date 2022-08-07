#include "d2pgo_test.hpp"
#include "posegraph_g2o.hpp"
#include "../src/d2pgo.h"
#include <thread>
#include <std_msgs/Int32.h>
#include <nav_msgs/Path.h>
// #include <visualization_msgs/Markers.h>

using namespace D2PGO;

class D2PGOTester {
    D2PGO::D2PGO * pgo = nullptr;
    std::string g2o_path;
    std::string solver_type;
    ros::Publisher dpgo_data_pub;
    ros::Subscriber dpgo_data_sub;
    bool is_4dof;
    std::thread th;
    std::string output_path;
    ros::NodeHandle & _nh;

    int max_steps = 100;

    std::map<int, ros::Publisher> path_pubs;
    std::vector<Swarm::LoopEdge> edges;
    std::map<FrameIdType, D2BaseFrame> keyframeid_agent_pose;
public:
    int self_id;
    void initSubandPub(ros::NodeHandle & nh) {
        dpgo_data_pub = nh.advertise<swarm_msgs::DPGOData>("/dpgo/pgo_data", 100);
        dpgo_data_sub = nh.subscribe("/dpgo/pgo_data", 100, &D2PGOTester::processDPGOData, this, ros::TransportHints().tcpNoDelay());
    }

    D2PGOTester(ros::NodeHandle & nh):
        _nh(nh) {
        nh.param<std::string>("g2o_path", g2o_path, "");
        nh.param<std::string>("output_path", output_path, "test.g2o");
        nh.param<int>("self_id", self_id, -1);
        nh.param<bool>("is_4dof", is_4dof, true);
        nh.param<std::string>("solver_type", solver_type, "arock");

        if (g2o_path != "")
            ROS_INFO("[D2PGO] agent %d parse g2o file: %s\n", self_id, g2o_path.c_str());
        else
            ROS_INFO("[D2PGO@%d] Need to indicate g2o path\n", self_id);
        read_g2o_agent(g2o_path, keyframeid_agent_pose, edges, is_4dof, self_id);
        ROS_INFO("[D2PGO@%d] Read %ld keyframes and %ld edges\n", self_id, keyframeid_agent_pose.size(), edges.size());

        D2PGOConfig config;
        config.self_id = self_id;
        if (is_4dof)
            config.pgo_pose_dof = PGO_POSE_4D;
        else
            config.pgo_pose_dof = PGO_POSE_6D;
        config.enable_ego_motion = false;
        config.ceres_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;// ceres::DENSE_SCHUR;
        config.ceres_options.num_threads = 1;
        config.ceres_options.max_num_iterations = 10000;
        config.ceres_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;// ceres::DOGLEG;
        nh.param<double>("max_solver_time", config.ceres_options.max_solver_time_in_seconds, 0.1);
        config.main_id = 0;
        config.arock_config.self_id = config.self_id;
        config.arock_config.verbose = true;
        config.arock_config.ceres_options = config.ceres_options;
        config.arock_config.max_steps = 1;
        nh.param<int>("max_steps", max_steps, 10);
        nh.param<double>("rho_frame_T", config.arock_config.rho_frame_T, 0.1);
        nh.param<double>("rho_frame_theta", config.arock_config.rho_frame_theta, 0.1);
        nh.param<double>("eta_k", config.arock_config.eta_k, 0.9);
        if (solver_type == "ceres") {
            config.mode = PGO_MODE_NON_DIST;
        } else {
            config.mode = PGO_MODE_DISTRIBUTED_AROCK;
        }

        pgo = new D2PGO::D2PGO(config);
        for (auto & kv : keyframeid_agent_pose) {
            pgo->addFrame(kv.second);
        }
        for (auto & edge : edges) {
            pgo->addLoop(edge, true); //In this test program, we use loop to initialize unknown poses.
        }

        pgo->bd_data_callback = [&] (const DPGOData & data) {
            // ROS_INFO("[D2PGO@%d] publish sync", self_id);
            dpgo_data_pub.publish(data.toROS());
        };

        pgo->postsolve_callback = [&] (void) {
            printf("Publish path\n");
            auto trajs = pgo->getOptimizedTrajs();
            pubTrajs(trajs);
            //Sleep for visualization.
            // usleep(100*1000);
        };
        initSubandPub(nh);
    }

    void processDPGOData(const swarm_msgs::DPGOData & data) {
        if (data.drone_id != self_id) {
            // ROS_INFO("[D2PGONode@%d] processDPGOData from drone %d", self_id, data.drone_id);
            pgo->inputDPGOData(DPGOData(data));
        }
    }

    void startSignalCallback(const std_msgs::Int32 & msg)  {
        startSolve();
    }

    void pubLoops(const std::vector<Swarm::LoopEdge> loops) {
        // for (auto loop : loops) {
        //     visualization_msgs::Marker marker;
        //     marker.ns = "marker";
        //     marker.id = loop.id;
        //     marker.type = visualization_msgs::Marker::LINE_STRIP;
        //     marker.action = visualization_msgs::Marker::ADD;
        //     marker.lifetime = ros::Duration();
        //     marker.scale.x = 0.02;
        //     marker.color.r = 1.0f;
        //     marker.color.a = 1.0;

        //     geometry_msgs::Point point0, point1;
        //     Eigen2Point(p0, point0);
        //     Eigen2Point(p1, point1);
        //     marker.points.push_back(point0);
        //     marker.points.push_back(point1);
        //     m_markers.push_back(marker);
        // }
    }

    void pubTrajs(const std::map<int, Swarm::DroneTrajectory> & trajs) {
        for (auto it : trajs) {
            auto drone_id = it.first;
            auto traj = it.second;
            if (path_pubs.find(drone_id) == path_pubs.end()) {
                path_pubs[drone_id] = _nh.advertise<nav_msgs::Path>("pgo_path_" + std::to_string(drone_id), 1000);
            }
            path_pubs[drone_id].publish(traj.get_ros_path());
        }
    }

    void startSolve() {
        th = std::thread([&]() {
            ROS_INFO("[D2PGO@%d] Start solve", self_id);
            for (int i = 0; i < max_steps; i ++) {
                pgo->solve(true);
            }
            ROS_INFO("[D2PGO@%d] End solve, writing reslts", self_id);
            //Write data
            writeDataG2o();
            ros::shutdown();
        });
    }

    void writeDataG2o() {
        auto local_frames = pgo->getAllLocalFrames();
        write_result_to_g2o(output_path, local_frames, edges);
        printf("[D2PGO@%d] Write result to %s\n", self_id, output_path.c_str());
    }


};

int main(int argc, char ** argv) {
    cv::setNumThreads(1);
    ros::init(argc, argv, "d2pgo_test");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    bool wait_for_start = false;
    n.param<bool>("wait_for_start", wait_for_start, "");
    D2PGOTester tester(n);
    ros::Subscriber start_sub;
    if (wait_for_start) {
        bool is_start = false;
        start_sub = n.subscribe("/dpgo/start_solve_trigger", 1, &D2PGOTester::startSignalCallback, &tester, ros::TransportHints().tcpNoDelay(true));
    } else {
        tester.startSolve();
    }
    printf("[D2PGO@%d] Waiting for solve\n", tester.self_id);
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    return 0;
}