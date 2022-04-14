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
        int init_imu_num = 10;
        Vector3d Gravity;
        MSCKFConfig():Gravity(0.0, 0.0, -g_norm) {
        }
    };

    struct IMUData {
        Vector3d acc;
        Vector3d gyro;
        IMUData(): acc(0.0, 0.0, 0.0),gyro(0.0, 0.0, 0.0){}
    };

    struct IMUBuffer
    {
        std::vector<IMUData> buf;
        void add(const IMUData & data) {
            buf.emplace_back(data);
        }

        Vector3d mean_acc() const {
            Vector3d acc_sum(0, 0, 0);
            for (auto & data : buf) {
                acc_sum += data.acc;
            }
            return acc_sum/size();
        }

        size_t size() const {
            return buf.size();
        }
    };
    
class MSCKF {
    MSCKFStateVector nominal_state;
    MSCKFErrorStateVector error_state;
    MSCKFConfig _config;
    double t_last = -1;
    bool initFirstPoseFlag = false;

    IMUBuffer imubuf;

    Eigen::Matrix<double, IMU_NOISE_DIM, IMU_NOISE_DIM> Q_imu;

public:
    MSCKF();
    void initFirstPose();
    void predict(const double t, const IMUData & imudata);
    void add_keyframe(const double t); //For convience, we require t here is exact same to last imu t
    void update(const D2Frontend::Feature & feature_by_id);
};
}