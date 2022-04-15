#include "d2vins/d2vins_types.hpp"
#include "d2vins/d2vins_params.hpp"
#include "feature_manager.hpp"
#include "d2vinsstate.hpp"

using namespace Eigen;
namespace D2VINS {

class D2Estimator {
protected:
    D2VINSConfig config;

    //Internal states
    bool initFirstPoseFlag = false;   
    D2EstimatorState state;
    IMUBuffer imubuf;
    Swarm::Pose last_pose; //Last pose;

    //Internal functions
    bool tryinitFirstPose(const D2Frontend::VisualImageDescArray & frame);
public:
    void inputImu(IMUData data);
    void inputImage(D2Frontend::VisualImageDescArray & frame);
    std::pair<double, Swarm::Pose> get_imu_propagation() const;
    std::pair<double, Swarm::Pose> get_odometry() const;
};
}