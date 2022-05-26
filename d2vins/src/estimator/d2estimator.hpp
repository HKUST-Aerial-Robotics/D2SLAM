#include "d2common/d2vinsframe.h"
#include "../d2vins_params.hpp"
#include "landmark_manager.hpp"
#include "d2vinsstate.hpp"
#include <swarm_msgs/Odometry.h>
#include <ceres/ceres.h>
#include "../visualization/visualization.hpp"

using namespace Eigen;
using D2Common::VisualImageDescArray;

namespace D2VINS {
class Marginalizer;
class D2Estimator {
protected:
    //Internal states
    bool initFirstPoseFlag = false;   
    D2EstimatorState state;
    IMUBuffer imubuf;
    std::map<int, IMUBuffer> remote_imu_bufs;
    Swarm::Odometry last_odom; //last accuacy odometry
    Swarm::Odometry last_prop_odom; //last imu propagation odometry
    Marginalizer * marginalizer = nullptr;
    //Internal functions
    bool tryinitFirstPose(VisualImageDescArray & frame);
    void addFrame(VisualImageDescArray & _frame);
    void addFrameRemote(const VisualImageDescArray & _frame);
    void solve();
    void setupImuFactors(ceres::Problem & problem);
    void setupLandmarkFactors(ceres::Problem & problem);
    void addIMUFactor(ceres::Problem & problem, FrameIdType frame_ida, FrameIdType frame_idb, IntegrationBase* _pre_integration);
    void setStateProperties(ceres::Problem & problem);
    void setupPriorFactor(ceres::Problem & problem);
    int frame_count = 0;
    D2Visualization visual;
    std::pair<bool, Swarm::Pose> initialFramePnP(const VisualImageDescArray & frame, 
        const Swarm::Pose & initial_pose);
    int solve_count = 0;
    int current_landmark_num = 0;
    ceres::Problem * problem = nullptr;
    std::set<int> used_camera_sets;
    std::vector<LandmarkPerId> margined_landmarks;
    int self_id;
    void addSldWinToFrame(VisualImageDescArray & frame);
    void addRemoteImuBuf(int drone_id, const IMUBuffer & imu_buf);
    bool isLocalFrame(FrameIdType frame_id) const;
public:
    D2Estimator(int drone_id);
    void inputImu(IMUData data);
    bool inputImage(VisualImageDescArray & frame);
    void inputRemoteImage(VisualImageDescArray & frame);
    Swarm::Odometry getImuPropagation() const;
    Swarm::Odometry getOdometry() const;
    void init(ros::NodeHandle & nh);
    D2EstimatorState & getState();
    std::vector<LandmarkPerId> getMarginedLandmarks() const;
};
}