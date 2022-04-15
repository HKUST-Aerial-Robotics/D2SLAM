#pragma once

#include <Eigen/Eigen>

using namespace Eigen;
namespace D2VINS {
struct D2VINSConfig {
    double acc_n = 0.1;
    double gyr_n = 0.05;
    double acc_w = 0.002;
    double gyr_w = 0.0004;
    double g_norm = 9.805;
    double IMU_FREQ = 400.0;
    int init_imu_num = 10;
    Vector3d Gravity;
    D2VINSConfig():Gravity(0.0, 0.0, g_norm) {
    }
};
}