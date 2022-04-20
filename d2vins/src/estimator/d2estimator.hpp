#include "d2vins/d2vins_types.hpp"
#include "d2vins/d2vins_params.hpp"
#include "landmark_manager.hpp"
#include "d2vinsstate.hpp"
#include <swarm_msgs/Odometry.h>
#include <ceres/ceres.h>
#include "visualization.hpp"

using namespace Eigen;

namespace D2VINS {
class D2Estimator {
protected:
    //Internal states
    bool initFirstPoseFlag = false;   
    D2EstimatorState state;
    IMUBuffer imubuf;
    Swarm::Odometry last_odom; //last accuacy odometry
    Swarm::Odometry last_prop_odom; //last imu propagation odometry

    //Internal functions
    bool tryinitFirstPose(const D2FrontEnd::VisualImageDescArray & frame);
    void addFrame(const D2FrontEnd::VisualImageDescArray & _frame);
    void solve();
    void setupImuFactors(ceres::Problem & problem);
    void setupLandmarkFactors(ceres::Problem & problem);
    void setStateProperties(ceres::Problem & problem);
    int frame_count = 0;
    D2Visualization visual;
public:
    void inputImu(IMUData data);
    void inputImage(D2FrontEnd::VisualImageDescArray & frame);
    Swarm::Odometry getImuPropagation() const;
    Swarm::Odometry getOdometry() const;
    void init(ros::NodeHandle & nh);
    D2EstimatorState & getState();
};
}