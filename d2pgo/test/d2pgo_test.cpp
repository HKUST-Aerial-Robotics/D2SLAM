#include "d2pgo_test.hpp"

#include <nav_msgs/Path.h>
#include <std_msgs/Int32.h>
#include <swarm_msgs/DPGOSignal.h>

#include <thread>

#include "../src/d2pgo.h"
#include "posegraph_g2o.hpp"
// #include <visualization_msgs/Markers.h>

using namespace D2PGO;

// #define BACKWARD_HAS_DW 1
// #include <backward.hpp>
// namespace backward
// {
//     backward::SignalHandling sh;
// }

class D2PGOTester {
  D2PGO::D2PGO* pgo = nullptr;
  std::string g2o_path;
  std::string solver_type;
  ros::Publisher dpgo_data_pub, dpgo_signal_pub;
  ros::Subscriber dpgo_data_sub, dpgo_signal_sub;
  bool is_4dof;
  std::thread th, th_process_delay;
  std::string output_path;
  ros::NodeHandle& _nh;
  bool multi = false;
  int max_steps = 100;
  int drone_num = 1;

  double simulate_delay_ms = 0;
  bool enable_simulate_delay = false;
  double max_solving_time = 10.0;
  D2PGOConfig config;

  std::map<int, ros::Publisher> path_pubs;
  std::vector<Swarm::LoopEdge> edges;
  std::map<FrameIdType, D2BaseFramePtr> keyframeid_agent_pose;
  std::map<double, D2Common::DPGOData> buf_for_simulate_delay;

 public:
  int self_id;
  void initSubandPub(ros::NodeHandle& nh) {
    dpgo_data_pub = nh.advertise<swarm_msgs::DPGOData>("/dpgo/pgo_data", 100);
    dpgo_signal_pub =
        nh.advertise<swarm_msgs::DPGOSignal>("/dpgo/pgo_signal", 100);
    dpgo_data_sub =
        nh.subscribe("/dpgo/pgo_data", 100, &D2PGOTester::processDPGOData, this,
                     ros::TransportHints().tcpNoDelay());
    dpgo_signal_sub =
        nh.subscribe("/dpgo/pgo_signal", 100, &D2PGOTester::processDPGOSignal,
                     this, ros::TransportHints().tcpNoDelay());
  }

  D2PGOTester(ros::NodeHandle& nh) : _nh(nh) {
    nh.param<std::string>("g2o_path", g2o_path, "");
    nh.param<std::string>("output_path", output_path, "test.g2o");
    nh.param<int>("self_id", self_id, -1);
    nh.param<bool>("is_4dof", is_4dof, true);
    nh.param<bool>("is_multi", multi, false);
    bool ignore_infor;
    nh.param<bool>("ignore_infor", ignore_infor, false);
    nh.param<std::string>("solver_type", solver_type, "arock");
    nh.param<double>("simulate_delay_ms", simulate_delay_ms, 0.0);
    nh.param<double>("max_solving_time", max_solving_time, 0.0);
    nh.param<int>("drone_num", drone_num, 1);
    if (simulate_delay_ms > 0) {
      enable_simulate_delay = true;
      th_process_delay = std::thread([&] { this->process_simulate_delay(); });
    }

    if (g2o_path != "")
      ROS_INFO("[D2PGO] agent %d parse g2o file: %s\n", self_id,
               g2o_path.c_str());
    else
      ROS_INFO("[D2PGO@%d] Need to indicate g2o path\n", self_id);
    read_g2o_agent(g2o_path, keyframeid_agent_pose, edges, is_4dof,
                   drone_num - 1, self_id, ignore_infor);
    ROS_INFO("[D2PGO@%d] Read %ld keyframes and %ld edges\n", self_id,
             keyframeid_agent_pose.size(), edges.size());

    config.self_id = self_id;
    if (is_4dof)
      config.pgo_pose_dof = PGO_POSE_4D;
    else
      config.pgo_pose_dof = PGO_POSE_6D;
    nh.param<double>("loop_distance_threshold", config.loop_distance_threshold,
                     1000);
    config.enable_ego_motion = false;
    config.ceres_options.linear_solver_type =
        ceres::SPARSE_NORMAL_CHOLESKY;  // ceres::DENSE_SCHUR;
    config.ceres_options.num_threads = 1;
    config.ceres_options.trust_region_strategy_type =
        ceres::LEVENBERG_MARQUARDT;  // ceres::DOGLEG;
    nh.param<double>("ceres_max_solver_time",
                     config.ceres_options.max_solver_time_in_seconds, 0.1);
    nh.param<int>("ceres_max_num_iterations",
                  config.ceres_options.max_num_iterations, 50);
    config.main_id = 0;
    config.arock_config.self_id = config.self_id;
    config.arock_config.verbose = true;
    config.arock_config.ceres_options = config.ceres_options;
    config.arock_config.max_steps = 1;
    config.g2o_output_path = output_path;
    config.write_g2o = false;
    nh.param<int>("max_steps", max_steps, 10);
    nh.param<double>("rho_frame_T", config.arock_config.rho_frame_T, 0.1);
    nh.param<double>("rho_frame_theta", config.arock_config.rho_frame_theta,
                     0.1);
    nh.param<double>("rho_rot_mat", config.arock_config.rho_rot_mat, 0.1);
    nh.param<double>("eta_k", config.arock_config.eta_k, 0.9);
    nh.param<bool>("enable_rot_init", config.enable_rotation_initialization,
                   true);
    nh.param<bool>("rot_init_enable_gravity_prior",
                   config.rot_init_config.enable_gravity_prior, true);
    nh.param<double>("rot_init_gravity_sqrt_info",
                     config.rot_init_config.gravity_sqrt_info, 10);
    nh.param<bool>("rot_init_enable_float32",
                   config.rot_init_config.enable_float32, false);
    nh.param<bool>("enable_linear_pose6d_solver",
                   config.rot_init_config.enable_pose6d_solver, false);
    nh.param<int>("linear_pose6d_iterations",
                  config.rot_init_config.pose6d_iterations, 10);
    nh.param<bool>("debug_rot_init_only", config.debug_rot_init_only, true);
    nh.param<double>("rot_init_state_eps", config.rot_init_state_eps, 0.01);
    config.rot_init_config.self_id = self_id;
    if (solver_type == "ceres") {
      config.mode = PGO_MODE_NON_DIST;
    } else {
      config.mode = PGO_MODE_DISTRIBUTED_AROCK;
    }

    pgo = new D2PGO::D2PGO(config);
    std::set<int> agent_ids;
    for (int i = 0; i < drone_num; i++) {
      agent_ids.insert(i);
    }
    pgo->setAvailableRobots(agent_ids);
    for (auto& kv : keyframeid_agent_pose) {
      pgo->addFrame(kv.second);
    }
    for (auto& edge : edges) {
      pgo->addLoop(edge, true);  // In this test program, we use loop to
                                 // initialize unknown poses.
    }

    pgo->bd_data_callback = [&](const DPGOData& data) {
      // ROS_INFO("[D2PGO@%d] publish sync", self_id);
      dpgo_data_pub.publish(data.toROS());
    };

    pgo->bd_signal_callback = [&](const std::string& signal) {
      // ROS_INFO("[D2PGO@%d] publish signal %s", self_id, signal.c_str());
      swarm_msgs::DPGOSignal msg;
      msg.header.stamp = ros::Time::now();
      msg.signal = signal;
      msg.drone_id = self_id;
      msg.target_id = -1;
      dpgo_signal_pub.publish(msg);
    };

    pgo->postsolve_callback = [&](void) {
      auto trajs = pgo->getOptimizedTrajs();
      pubTrajs(trajs);
      // Sleep for visualization.
      // usleep(100*1000);
    };
    initSubandPub(nh);
  }

  void processDPGOData(const swarm_msgs::DPGOData& data) {
    if (data.drone_id != self_id) {
      // ROS_INFO("[D2PGONode@%d] processDPGOData from drone %d", self_id,
      // data.drone_id);
      if (enable_simulate_delay) {
        ros::Time now = ros::Time::now();
        buf_for_simulate_delay[now.toSec()] = data;
      } else {
        pgo->inputDPGOData(DPGOData(data));
      }
    }
  }

  void processDPGOSignal(const swarm_msgs::DPGOSignal& msg) {
    if (msg.drone_id != self_id) {
      ROS_INFO("[D2PGONode@%d] processDPGOSignal from drone %d: %s", self_id,
               msg.drone_id, msg.signal.c_str());
      pgo->inputDPGOsignal(msg.drone_id, msg.signal);
    }
  }

  void startSignalCallback(const std_msgs::Int32& msg) { startSolve(); }

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

  void pubTrajs(const std::map<int, Swarm::DroneTrajectory>& trajs) {
    for (auto it : trajs) {
      auto drone_id = it.first;
      auto traj = it.second;
      if (path_pubs.find(drone_id) == path_pubs.end()) {
        path_pubs[drone_id] = _nh.advertise<nav_msgs::Path>(
            "pgo_path_" + std::to_string(drone_id), 1000);
      }
      path_pubs[drone_id].publish(traj.get_ros_path());
    }
  }

  void startSolve() {
    th = std::thread([&]() {
      Utility::TicToc t_solve;
      int iter = 0;
      for (int i = 0; i < max_steps; i++) {
        iter++;
        if (multi) {
          pgo->solve_multi(true);
        } else {
          pgo->solve_single();
        }
        // usleep(20*1000);
        if (t_solve.toc() / 1000.0 > max_solving_time) {
          printf("[D2PGO%d] Solve timeout. Time: %fms\n", self_id,
                 t_solve.toc());
          break;
        }
      }
      printf("[D2PGO%d] Solve done. Time: %fms iters %d\n", self_id,
             t_solve.toc(), iter);
      // Write data
      if (config.perturb_mode) {
        pgo->postPerturbSolve();
      }
      writeDataG2o();
      printf("[D2PGO%d] Write data done. Finish solve.\n", self_id);
      fflush(stdout);
      // ros::shutdown();
      exit(0);
    });
  }

  void writeDataG2o() {
    auto local_frames = pgo->getAllLocalFrames();
    write_result_to_g2o(output_path, local_frames, edges);
    printf("[D2PGO@%d] Write result to %s\n", self_id, output_path.c_str());
  }

  void process_simulate_delay() {
    while (ros::ok()) {
      ros::Time now = ros::Time::now();
      double now_sec = now.toSec();
      for (auto it = buf_for_simulate_delay.begin();
           it != buf_for_simulate_delay.end();) {
        if (now_sec - it->first > simulate_delay_ms / 1000.0) {
          pgo->inputDPGOData(DPGOData(it->second));
          it = buf_for_simulate_delay.erase(it);
        } else {
          it++;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
};

int main(int argc, char** argv) {
  cv::setNumThreads(1);
  ros::init(argc, argv, "d2pgo_test");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);
  bool wait_for_start = false;
  n.param<bool>("wait_for_start", wait_for_start, "");
  D2PGOTester tester(n);
  ros::Subscriber start_sub;
  if (wait_for_start) {
    bool is_start = false;
    start_sub = n.subscribe("/dpgo/start_solve_trigger", 1,
                            &D2PGOTester::startSignalCallback, &tester,
                            ros::TransportHints().tcpNoDelay(true));
  } else {
    tester.startSolve();
  }
  printf("[D2PGO@%d] Waiting for solve\n", tester.self_id);
  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();
  return 0;
}