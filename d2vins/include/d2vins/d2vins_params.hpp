#pragma once
#include <Eigen/Eigen>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <ceres/ceres.h>

using namespace Eigen;

const Vector3d Gravity = Vector3d(0, 0, 9.805);

#define POSE_SIZE 7
#define FRAME_SPDBIAS_SIZE 9
#define TD_SIZE 1
#define INV_DEP_SIZE 1
#define POS_SIZE 3

typedef double state_type;

namespace D2VINS {
struct D2VINSConfig {
    double acc_n = 0.1;
    double gyr_n = 0.05;
    double acc_w = 0.002;
    double gyr_w = 0.0004;
    // double g_norm = 9.805;
    double IMU_FREQ = 400.0;
    int init_imu_num = 10;
    int max_sld_win_size = 10;
    int landmark_estimate_tracks = 4; //thres for landmark to tracking
    double focal_length = 460.0;
    bool estimate_td = false;
    bool estimate_extrinsic = false;
    int camera_num = 1; // number of cameras;
    double solver_time = 0.04;
    enum {
        INIT_POSE_IMU,
        INIT_POSE_PNP
    } init_method = INIT_POSE_IMU;

    bool verbose = true;

    enum {
        LM_INV_DEP,
        LM_POS
    } landmark_param = LM_INV_DEP;

    std::vector<Swarm::Pose> camera_extrinsics;
    double td_initial = 0.0;
    ceres::Solver::Options options;

    D2VINSConfig() {
    }
    void init() {
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.num_threads = 1;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_solver_time_in_seconds = solver_time;
    }
};

extern D2VINSConfig * params;
void initParams(ros::NodeHandle & nh);
}