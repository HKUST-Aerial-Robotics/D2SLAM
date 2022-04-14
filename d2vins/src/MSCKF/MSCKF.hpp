#include "MSCKF_state.hpp"
#include "d2frontend/d2frontend_types.h"

namespace D2VINS {
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
    void update(const FeaturePerId & feature_by_id);
};
}