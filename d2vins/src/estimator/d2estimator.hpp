#include "d2vins/d2vins_types.hpp"
#include "d2vins/d2vins_params.hpp"
#include "landmark_manager.hpp"
#include "d2vinsstate.hpp"
#include <swarm_msgs/Odometry.h>
#include <ceres/ceres.h>
#include "visualization.hpp"

using namespace Eigen;
using D2FrontEnd::VisualImageDescArray;

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
    bool tryinitFirstPose(const VisualImageDescArray & frame);
    void addFrame(const VisualImageDescArray & _frame);
    void solve();
    void setupImuFactors(ceres::Problem & problem);
    void setupLandmarkFactors(ceres::Problem & problem);
    void setStateProperties(ceres::Problem & problem);
    int frame_count = 0;
    D2Visualization visual;
    std::pair<bool, Swarm::Pose> initialFramePnP(const VisualImageDescArray & frame, 
        const Swarm::Pose & pose);
    int solve_count = 0;
    int current_landmark_num = 0;
public:
    D2Estimator() {}
    void inputImu(IMUData data);
    void inputImage(VisualImageDescArray & frame);
    Swarm::Odometry getImuPropagation() const;
    Swarm::Odometry getOdometry() const;
    void init(ros::NodeHandle & nh);
    D2EstimatorState & getState();
};
}