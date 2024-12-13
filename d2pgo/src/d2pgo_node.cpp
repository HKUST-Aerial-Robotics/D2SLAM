#include <ros/ros.h>

#include "d2pgo.h"
#include "geometry_msgs/PoseStamped.h"
#include "swarm_msgs/ImageArrayDescriptor.h"
#include "swarm_msgs/swarm_fused.h"

#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

namespace D2PGO {
class D2PGONode {
  D2PGO *pgo = nullptr;
  ros::Subscriber frame_sub, remote_frame_sub, loop_sub, dpgo_data_sub;
  ros::Timer solver_timer;
  double solver_timer_freq = 10;
  D2PGOConfig config;
  std::map<int, ros::Publisher> path_pubs;
  std::map<int, ros::Publisher> odom_pubs;
  ros::Publisher path_pub;
  ros::Publisher dpgo_data_pub;
  ros::Publisher drone_traj_pub, swarm_fused_pub;
  bool write_to_file = true;
  bool multi = false;
  int write_to_file_step = 5;
  int pub_count = 0;
  std::string output_folder;
  ros::NodeHandle *_nh;

 protected:
  void processImageArray(const swarm_msgs::VIOFrame &vioframe) {
    if (vioframe.is_keyframe) {
      auto frame = std::make_shared<D2BaseFrame>(vioframe);
      pgo->addFrame(frame);
    }
  }

  void processLoop(const swarm_msgs::LoopEdge &loop_info) {
    // ROS_INFO("[D2PGONode@%d] processLoop from %ld to %ld", config.self_id,
    // loop_info.keyframe_id_a, loop_info.keyframe_id_b);
    pgo->addLoop(Swarm::LoopEdge(loop_info));
  }

  void processDPGOData(const swarm_msgs::DPGOData &data) {
    if (data.drone_id != config.self_id) {
      // ROS_INFO("[D2PGONode@%d] processDPGOData from drone %d",
      // config.self_id, data.drone_id);
      pgo->inputDPGOData(DPGOData(data));
    }
  }

  void pubTrajs(std::map<int, Swarm::DroneTrajectory> &trajs) {
    for (auto it : trajs) {
      auto drone_id = it.first;
      auto traj = it.second;
      if (path_pubs.find(drone_id) == path_pubs.end()) {
        path_pubs[drone_id] = _nh->advertise<nav_msgs::Path>(
            "pgo_path_" + std::to_string(drone_id), 1000);
      }
      if (drone_id == config.self_id) {
        path_pub.publish(traj.get_ros_path());
      }
      drone_traj_pub.publish(traj.toRos());
      path_pubs[drone_id].publish(traj.get_ros_path());
      if (write_to_file && pub_count % write_to_file_step == 0) {
        std::ofstream csv(
            output_folder + "/pgo_" + std::to_string(drone_id) + ".csv",
            std::ios::out);
        for (size_t i = 0; i < traj.trajectory_size(); i++) {
          double stamp = traj.stamp_by_index(i);
          auto pose = traj.pose_by_index(i);
          csv << std::setprecision(std::numeric_limits<long double>::digits10 +
                                   1)
              << stamp << " " << pose.toStr(true) << std::endl;
        }
        csv.close();
      }
    }
    // printf("[D2PGONode@%d] pubTrajs, %ld trajs\n", config.self_id,
    // trajs.size());
    pub_count++;
  }

  void pubLatestPredictResults() {
    swarm_msgs::swarm_fused swarm_fused;
    swarm_fused.header.stamp = ros::Time::now();
    swarm_fused.self_id = config.self_id;
    swarm_fused.reference_frame_id = pgo->getReferenceFrameId();
    auto latest_odoms = pgo->getPredictedOdoms();
    // printf("[D2PGONode@%d] pubLatestPredictResults, %ld odoms\n",
    // config.self_id, latest_odoms.size());
    for (auto it : latest_odoms) {
      auto drone_id = it.first;
      auto pose = it.second.pose();
      auto pose_ros = pose.toROS();
      swarm_fused.ids.emplace_back(drone_id);
      swarm_fused.local_drone_position.emplace_back(pose_ros.position);
      swarm_fused.local_drone_rotation.emplace_back(pose_ros.orientation);
      swarm_fused.local_drone_yaw.emplace_back(pose.yaw());
      if (drone_id == config.self_id) {
        swarm_fused.self_pos = pose_ros.position;
        swarm_fused.self_yaw = pose.yaw();
      }
      if (odom_pubs.find(drone_id) == odom_pubs.end()) {
        odom_pubs[drone_id] = _nh->advertise<geometry_msgs::PoseStamped>(
            "pose_" + std::to_string(drone_id), 1000);
      }
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.stamp = ros::Time(it.second.stamp);
      pose_stamped.header.frame_id = "world";
      pose_stamped.pose = pose_ros;
      odom_pubs[drone_id].publish(pose_stamped);
    }
    swarm_fused_pub.publish(swarm_fused);
  }

  void solverTimerCallback(const ros::TimerEvent &event) {
    bool succ;
    if (multi) {
      // printf("[D2PGO] try to solve multi......\n");
      succ = pgo->solve_multi();
    } else {
      succ = pgo->solve_single();
    }
    if (succ) {
      auto trajs = pgo->getOptimizedTrajs();
      pubTrajs(trajs);
    }
    pubLatestPredictResults();
  }

  void Init(ros::NodeHandle &nh) {
    InitParams(nh);
    _nh = &nh;
    pgo = new D2PGO(config);
    path_pub = _nh->advertise<nav_msgs::Path>("pgo_path", 1000);
    drone_traj_pub = _nh->advertise<swarm_msgs::DroneTraj>("pgo_traj", 1000);
    dpgo_data_pub = _nh->advertise<swarm_msgs::DPGOData>("pgo_data", 1000);
    swarm_fused_pub =
        _nh->advertise<swarm_msgs::swarm_fused>("swarm_fused", 1000);
    pgo->bd_data_callback = [&](const DPGOData &data) {
      dpgo_data_pub.publish(data.toROS());
    };
    dpgo_data_sub = nh.subscribe("pgo_data", 1000, &D2PGONode::processDPGOData,
                                 this, ros::TransportHints().tcpNoDelay());
    frame_sub = nh.subscribe("frame_local", 1000, &D2PGONode::processImageArray,
                             this, ros::TransportHints().tcpNoDelay());
    remote_frame_sub =
        nh.subscribe("frame_remote", 1000, &D2PGONode::processImageArray, this,
                     ros::TransportHints().tcpNoDelay());
    loop_sub = nh.subscribe("loop", 1000, &D2PGONode::processLoop, this,
                            ros::TransportHints().tcpNoDelay());
    solver_timer = nh.createTimer(ros::Duration(1.0 / solver_timer_freq),
                                  &D2PGONode::solverTimerCallback, this);
    printf("[D2PGONode@%d] Initialized\n", config.self_id);
  }

  void InitParams(ros::NodeHandle &nh) {
    std::string vins_config_path;
    nh.param<std::string>("vins_config_path", vins_config_path, "");
    cv::FileStorage fsSettings;
    try {
      fsSettings.open(vins_config_path.c_str(), cv::FileStorage::READ);
      std::cout << "PGO Loaded VINS config from " << vins_config_path
                << std::endl;
    } catch (cv::Exception ex) {
      std::cerr << "ERROR:" << ex.what() << " Can't open config file"
                << std::endl;
      exit(-1);
    }
    fsSettings["output_path"] >> output_folder;
    config.write_g2o = (int)fsSettings["write_g2o"];
    fsSettings["g2o_output_path"] >> config.g2o_output_path;
    write_to_file = (int)fsSettings["write_pgo_to_file"];
    config.g2o_output_path = output_folder + "/";  // + config.g2o_output_path;
    config.mode = static_cast<PGO_MODE>((int)fsSettings["pgo_mode"]);
    nh.param<int>("self_id", config.self_id, -1);
    bool is_4dof;
    nh.param<bool>("is_4dof", is_4dof, true);
    if (is_4dof) {
      config.pgo_pose_dof = PGO_POSE_4D;
    } else {
      config.pgo_pose_dof = PGO_POSE_6D;
    }
    config.pcm_rej.is_4dof = is_4dof;

    // Config ceres
    config.ceres_options.linear_solver_type =
        ceres::SPARSE_NORMAL_CHOLESKY;  // ceres::DENSE_SCHUR;
    config.ceres_options.num_threads = 1;
    config.ceres_options.trust_region_strategy_type =
        ceres::LEVENBERG_MARQUARDT;  // ceres::DOGLEG;
    config.ceres_options.max_solver_time_in_seconds =
        fsSettings["pgo_solver_time"];
    config.main_id = 1;

    // Config arock
    config.arock_config.self_id = config.self_id;
    config.arock_config.ceres_options = config.ceres_options;
    config.arock_config.rho_frame_T = fsSettings["pgo_rho_frame_T"];
    config.arock_config.rho_frame_theta = fsSettings["pgo_rho_frame_theta"];
    config.arock_config.eta_k = fsSettings["pgo_eta_k"];
    config.arock_config.max_steps = 1;

    // Outlier rejection
    config.is_realtime = true;
    config.enable_pcm = (int)fsSettings["enable_pcm"];
    config.pcm_rej.pcm_thres = fsSettings["pcm_thres"];
    config.enable_rotation_initialization = false;
    config.enable_gravity_prior = (int)fsSettings["enable_gravity_prior"];
    config.rot_init_config.gravity_sqrt_info = fsSettings["gravity_sqrt_info"];
    solver_timer_freq = (double)fsSettings["solver_timer_freq"];
    config.perturb_mode = true;
    // Debugging
    config.debug_save_g2o_only = (int)fsSettings["debug_save_g2o_only"];
    config.loop_distance_threshold = (int)fsSettings["accept_loop_max_pos"];
    if (config.mode == PGO_MODE::PGO_MODE_NON_DIST) {
      multi = false;
      printf("[D2PGO] In single mode enable_pcm %d pcm_thres %.1f\n",
             config.enable_pcm, config.pcm_rej.pcm_thres);
    } else {
      multi = true;
      printf("[D2PGO] In multi mode enable_pcm %d pcm_thres %.1f\n",
             config.enable_pcm, config.pcm_rej.pcm_thres);
    }
  }

 public:
  D2PGONode(ros::NodeHandle &nh) { Init(nh); }
};
}  // namespace D2PGO

int main(int argc, char **argv) {
  cv::setNumThreads(1);
  ros::init(argc, argv, "d2pgo");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);

  D2PGO::D2PGONode d2pgonode(n);
  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();
  return 0;
}