#include "feature_manager.hpp"
#include "d2vins_types.hpp"

using namespace Eigen;
namespace D2VINS {
struct VINSFrame {
    double t;
    int frame_id;
    Vector3d Ps;
    Vector3d Vs;
    Matrix3d Rs;
    Vector3d Bas;
    Vector3d Bgs;
}

class SlidingWindow {
protected:
    std::vector<VINSFrame> sld_win;
public:

}

class D2Estimator {
protected:
    D2VINSConfig config;

    //Internal states
    bool initFirstPoseFlag = false;
    SlidingWindow sld_win;
    double td = 0.0; //estimated td;

    IMUBuffer imu_buf;
    Swarm::Pose last_pose; //Last pose;

    //Internal functions
    bool tryinitFirstPose();
public:
    void inputImu(IMUData data);
    void inputImage(VisualImageDescArray & frame);
    std::pair<double, Swarm::Pose> get_imu_propagation() const;
    std::pair<double, Swarm::Pose> get_odometry() const;
};
}