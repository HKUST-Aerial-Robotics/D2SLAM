#include "d2pgo_test.hpp"
#include "posegraph_g2o.hpp"
#include "../src/d2pgo.h"
#include <thread>
#include <std_msgs/Int32.h>

using namespace D2PGO;

class D2PGOTester {
    D2PGO::D2PGO * pgo = nullptr;
    std::string g2o_path;
    std::string solver_type;
    ros::Publisher dpgo_data_pub;
    ros::Subscriber dpgo_data_sub;
    bool is_4dof;
    std::thread th;
public:
    int self_id;
    void initSubandPub(ros::NodeHandle & nh) {
        dpgo_data_pub = nh.advertise<swarm_msgs::DPGOData>("/dpgo/pgo_data", 100);
        dpgo_data_sub = nh.subscribe("/dpgo/pgo_data", 100, &D2PGOTester::processDPGOData, this, ros::TransportHints().tcpNoDelay());
    }

    D2PGOTester(ros::NodeHandle & nh) {
        nh.param<std::string>("g2o_path", g2o_path, "");
        nh.param<int>("self_id", self_id, -1);
        nh.param<bool>("is_4dof", is_4dof, true);
        nh.param<std::string>("solver_type", solver_type, "arock");

        if (g2o_path != "")
            ROS_INFO("[D2PGO] agent %d parse g2o file: %s\n", self_id, g2o_path.c_str());
        else
            ROS_INFO("[D2PGO@%d] Need to indicate g2o path\n", self_id);
        std::map<FrameIdType, D2BaseFrame> keyframeid_agent_pose;
        std::vector<Swarm::LoopEdge> edges;
        read_g2o_agent(g2o_path, keyframeid_agent_pose, edges, is_4dof, self_id);
        ROS_INFO("[D2PGO@%d] Read %ld keyframes and %ld edges\n", self_id, keyframeid_agent_pose.size(), edges.size());

        D2PGOConfig config;
        config.self_id = self_id;
        config.enable_ego_motion = false;
        config.ceres_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;// ceres::DENSE_SCHUR;
        config.ceres_options.num_threads = 1;
        config.ceres_options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;// ceres::DOGLEG;
        nh.param<double>("max_solver_time", config.ceres_options.max_solver_time_in_seconds, 0.1);
        config.main_id = 0;
        config.arock_config.self_id = config.self_id;
        config.arock_config.ceres_options = config.ceres_options;
        nh.param<int>("max_steps", config.arock_config.max_steps, 10);
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

    void startSolve() {
        th = std::thread([&]() {
            ROS_INFO("[D2PGO@%d] Start solve", self_id);
            pgo->solve();
            ROS_INFO("[D2PGO@%d] End solve", self_id);
            // ros::shutdown();
        });
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