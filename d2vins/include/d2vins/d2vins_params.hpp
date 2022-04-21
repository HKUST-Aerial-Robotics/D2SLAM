#pragma once
#include <Eigen/Eigen>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <ceres/ceres.h>

using namespace Eigen;

#define POSE_SIZE 7
#define FRAME_SPDBIAS_SIZE 9
#define TD_SIZE 1
#define INV_DEP_SIZE 1
#define POS_SIZE 3


namespace D2VINS {

typedef double state_type;
extern Vector3d Gravity;

struct D2VINSConfig {
    double acc_n = 0.1;
    double gyr_n = 0.05;
    double acc_w = 0.002;
    double gyr_w = 0.0004;
    // double g_norm = 9.805;
    double IMU_FREQ = 400.0;
    int init_imu_num = 10;
    int max_sld_win_size = 100;
    int landmark_estimate_tracks = 4; //thres for landmark to tracking
    double focal_length = 460.0;
    bool estimate_td = false;
    bool estimate_extrinsic = false;
    int camera_num = 1; // number of cameras;
    int min_solve_frames = 9;
    int frame_step = 3; //step of frame to use in backend.
    double solver_time = 0.04;
    double min_inv_dep = 1e-1; //10 meter away
    double depth_sqrt_inf = 20.0;
    int pnp_min_inliers = 8;
    int pnp_iteratives = 100;
    bool debug_print_states = false;
    bool fuse_dep = true;
    double max_depth_to_fuse = 5.;
    double td_max_diff = 0.0025;
    std::string output_folder;
    enum InitialMethod {
        INIT_POSE_IMU,
        INIT_POSE_PNP
    } init_method = INIT_POSE_PNP;

    bool verbose = true;

    enum {
        LM_INV_DEP,
        LM_POS
    } landmark_param = LM_INV_DEP;

    std::vector<Swarm::Pose> camera_extrinsics;
    double td_initial = 0.0;
    ceres::Solver::Options options;
    void init(const std::string & config_file);

};

extern D2VINSConfig * params;
void initParams(ros::NodeHandle & nh);
}