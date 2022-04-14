#include "MSCKF_state.hpp"
#include "d2frontend/d2frontend_types.h"

namespace D2VINS {
    struct MSCKFConfig {
        double acc_n = 0.1;
        double gyr_n = 0.05;
        double acc_w = 0.002;
        double gyr_w = 0.0004;
        double g_norm = 9.805;
        double IMU_FREQ = 400.0;
        Vector3d Gravity;
        MSCKFConfig():Gravity(0.0, 0.0, -g_norm) {
        }
    };
class MSCKF {
    MSCKFStateVector nominal_state;
    MSCKFErrorStateVector error_state;
    MSCKFConfig _config;
    double t_last = -1;

    Eigen::Matrix<double, IMU_NOISE_DIM, IMU_NOISE_DIM> Q_imu;

public:
    MSCKF();

    void predict(const double t, Vector3d acc, Vector3d gyro);
    void add_keyframe(const double t); //For convience, we require t here is exact same to last imu t
    void update(const D2Frontend::Feature & feature_by_id);
};
}