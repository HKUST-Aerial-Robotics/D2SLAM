#include "d2vins/d2vins_types.hpp"
#include "d2vins/d2vins_params.hpp"
#include "landmark_manager.hpp"
#include "d2vinsstate.hpp"
#include <swarm_msgs/Odometry.h>

using namespace Eigen;
namespace D2VINS {

class D2Estimator {
protected:
    D2VINSConfig config;

    //Internal states
    bool initFirstPoseFlag = false;   
    D2EstimatorState state;
    IMUBuffer imubuf;
    Swarm::Odometry last_odom; //last accuacy odometry
    Swarm::Odometry last_prop_odom; //last imu propagation odometry

    //Internal functions
    bool tryinitFirstPose(const D2FrontEnd::VisualImageDescArray & frame);
    VINSFrame initFrame(const D2FrontEnd::VisualImageDescArray & _frame);
public:
    void inputImu(IMUData data);
    void inputImage(D2FrontEnd::VisualImageDescArray & frame);
    Swarm::Odometry getImuPropagation() const;
    Swarm::Odometry getOdometry() const;
};
}