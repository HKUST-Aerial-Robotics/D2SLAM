#pragma once
#include <Eigen/Eigen>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <ceres/ceres.h>
#include <d2common/d2basetypes.h>

#define UNIT_SPHERE_ERROR
using namespace Eigen;

namespace D2Common {
struct ConsensusSolverConfig;
};
namespace D2VINS {
using D2Common::CameraConfig;
using D2Common::ESTIMATION_MODE;
struct D2VINSConfig {
    //Inputs
    std::string imu_topic;

    //Sensor config
    double acc_n = 0.1;
    double gyr_n = 0.05;
    double acc_w = 0.002;
    double gyr_w = 0.0004;
    double focal_length = 460.0;
    double initial_pos_sqrt_info = 1000.0;
    double initial_vel_sqrt_info = 100.0;
    double initial_ba_sqrt_info = 5.0;
    double initial_bg_sqrt_info = 10.0;
    double initial_yaw_sqrt_info = 10000.0;
    double initial_cam_pos_sqrt_info = 1000.0;
    double initial_cam_ang_sqrt_info = 10000.0;
    D2Common::CameraConfig camera_configuration;
    bool enable_loop;

    //Sensor frequency
    double IMU_FREQ = 400.0;
    double IMAGE_FREQ = 20.0;
    int camera_num = 1; // number of cameras;
    int frame_step = 3; //step of frame to use in backend.
    
    //Sliding window
    int min_solve_frames = 9;
    int max_sld_win_size = 10;
    int landmark_estimate_tracks = 4; //thres for landmark to tracking

    //Initialization
    enum InitialMethod {
        INIT_POSE_IMU,
        INIT_POSE_PNP
    };
    int pnp_min_inliers = 20;
    int pnp_iteratives = 100;
    int init_imu_num = 10;
    InitialMethod init_method = INIT_POSE_PNP;
    double depth_estimate_baseline = 0.05;
    double tri_max_err = 0.1;
    double mono_initial_tri_max_err = 0.05;
    bool add_vel_ba_prior = false;
    int solve_relative_pose_min_pts = 20;
    double solve_relative_pose_min_parallex = 30.0/460.0;
    bool enable_sfm_initialization = false;
    double init_acc_bias_threshold = 0.2;
    
    //Estimation
    bool estimate_td = false;
    bool estimate_extrinsic = false;
    double solver_time = 0.04;
    enum {
        LM_INV_DEP,
        LM_POS
    } landmark_param = LM_INV_DEP;
    bool always_fixed_first_pose = false;
    double process_input_timer = 100.0;
    double estimator_timer_freq = 100.0;
    int warn_pending_frames = 10;
    ESTIMATION_MODE estimation_mode;
    double estimate_extrinsic_vel_thres = 0.2;
    int max_solve_cnt = 10000;
    int max_solve_measurements = -1;
    int min_solve_cnt = 10;
    bool not_estimate_first_extrinsic = false;

    //Fuse depth
    bool fuse_dep = true;
    double max_inv_dep = 1e1; //10 cm away
    double default_inv_dep = 1e-1;
    double depth_sqrt_inf = 20.0;
    double max_depth_to_fuse = 5.;
    double min_depth_to_fuse = 0.3;

    //Solver
    ceres::Solver::Options ceres_options;
    D2Common::ConsensusSolverConfig * consensus_config = nullptr;
    bool consensus_sync_to_start = true;
    int consensus_trigger_time_err_us = 50;
    double wait_for_start_timout = 300.0;

    //Outlier rejection
    int perform_outlier_rejection_num = 50;
    double landmark_outlier_threshold = 10.0;
    double remove_scale_outlier_threshold = 10.0; // Remove landmark with scale remove_scale_outlier_threshold * middle scale

    //Margin config
    bool margin_sparse_solver = true;
    bool enable_marginalization = true;
    int remove_base_when_margin_remote = 2;
    bool margin_enable_fej = true;

    //Safety
    int min_measurements_per_keyframe = 10;
    double max_imu_time_err = 0.0025;

    //Multi-drone
    int self_id = 0;
    int main_id = 0; // This main id is whom will be base reference frame so has a huge prior.
    double nearby_drone_dist = 5.0;
    double nearby_drone_yaw_dist = 1000; //Degree
    bool lazy_broadcast_keyframe = false;

    //Debug
    bool debug_print_states = false;
    bool debug_print_sldwin = false;
    std::string output_folder;
    bool enable_perf_output = false;
    bool debug_write_margin_matrix = false;
    bool pub_visual_frame = false;

    bool verbose = true;
    bool print_network_status = false;

    //Initialial states
    std::vector<Swarm::Pose> camera_extrinsics;
    double td_initial = 0.0;
    
    //Comm
    std::string lcm_uri;

    void init(const std::string & config_file);
};

extern D2VINSConfig * params;
void initParams(ros::NodeHandle & nh);
}
