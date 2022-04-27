#pragma once
#include <Eigen/Eigen>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <ceres/ceres.h>

using namespace Eigen;

#define POSE_SIZE 7
#define POSE_EFF_SIZE 6
#define FRAME_SPDBIAS_SIZE 9
#define TD_SIZE 1
#define INV_DEP_SIZE 1
#define POS_SIZE 3


namespace D2VINS {

typedef double state_type;
extern Vector3d Gravity;

struct D2VINSConfig {
    //Sensor config
    double acc_n = 0.1;
    double gyr_n = 0.05;
    double acc_w = 0.002;
    double gyr_w = 0.0004;
    double focal_length = 460.0;
    
    //Sensor frequency
    double IMU_FREQ = 400.0;
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
    int pnp_min_inliers = 8;
    int pnp_iteratives = 100;
    int init_imu_num = 10;
    InitialMethod init_method = INIT_POSE_PNP;
    
    //Estimation
    bool estimate_td = false;
    bool estimate_extrinsic = false;
    double solver_time = 0.04;
    enum {
        LM_INV_DEP,
        LM_POS
    } landmark_param = LM_INV_DEP;
    //Fuse depth
    bool fuse_dep = true;
    double min_inv_dep = 1e-1; //10 meter away
    double depth_sqrt_inf = 20.0;
    double max_depth_to_fuse = 5.;
    double min_depth_to_fuse = 0.3;
    ceres::Solver::Options options;

    //Outlier rejection
    int perform_outlier_rejection_num = 50;
    double landmark_outlier_threshold = 10.0;

    //Margin config
    bool margin_sparse_solver = true;
    bool enable_marginalization = true;
    bool use_llt_for_decompose_A_b = false; //After schur complement, use LLT to decompose A and b.

    //Safety
    int min_measurements_per_keyframe = 10;
    double max_imu_time_err = 0.0025;

    //Debug
    bool debug_print_states = false;
    std::string output_folder;
    bool debug_print_solver_details = false;
    bool debug_print_marginal = false;

    bool verbose = true;

    //Initialial states
    std::vector<Swarm::Pose> camera_extrinsics;
    double td_initial = 0.0;

    void init(const std::string & config_file);
};

extern D2VINSConfig * params;
void initParams(ros::NodeHandle & nh);
}