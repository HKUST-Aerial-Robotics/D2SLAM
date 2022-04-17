#include "d2vins/d2vins_types.hpp"
#include "d2vins/d2vins_params.hpp"
#include "landmark_manager.hpp"
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
    double last_pose_t = 0.0;

    Swarm::Pose last_propagation_pose; //Last pose;
    double last_propagation_t = 0.0;

    //Internal functions
    bool tryinitFirstPose(const D2Frontend::VisualImageDescArray & frame);
    VINSFrame initFrame(const D2Frontend::VisualImageDescArray & _frame);
public:
    void inputImu(IMUData data);
    void inputImage(D2Frontend::VisualImageDescArray & frame);
    std::pair<double, Swarm::Pose> getImuPropagation() const;
    std::pair<double, Swarm::Pose> getOdometry() const;
};
}