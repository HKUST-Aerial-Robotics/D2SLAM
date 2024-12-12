#include "d2common/d2vinsframe.h"
#include "../d2vins_params.hpp"
#include "landmark_manager.hpp"
#include "d2vinsstate.hpp"
#include <swarm_msgs/Odometry.h>
#include <ceres/ceres.h>
#include "../visualization/visualization.hpp"
#include <d2common/solver/SolverWrapper.hpp>
#include "solver/ConsensusSync.hpp"
#include <mutex>

using namespace Eigen;
using D2Common::VisualImageDescArray;

namespace D2VINS {
class Marginalizer;
class D2VINSNet;
struct DistributedVinsData;

enum SyncSignal {
    DSolverAbort = 0,
    DSolverReady,
    DSolverStart,
    DSolverNonDist
};

class D2Estimator {
protected:
    //Internal states
    bool initFirstPoseFlag = false;   
    D2EstimatorState state;
    std::map<int, IMUBuffer> imu_bufs;
    std::map<int, Swarm::Odometry> last_prop_odom; //last imu propagation odometry
    std::map<int, Swarm::Pose> last_pgo_poses; //last pgo poses
    Marginalizer * marginalizer = nullptr;
    SolverWrapper * solver = nullptr;
    D2VINSNet * vinsnet = nullptr;
    int solve_count = 0;
    int current_landmark_num = 0;
    int current_measurement_num = 0;
    std::vector<LandmarkPerId> margined_landmarks;
    int self_id;
    int frame_count = 0;
    int64_t solve_token = 0;
    D2Visualization visual;
    std::set<int> ready_drones;
    bool ready_to_start = false;
    std::map<FrameIdType, int> keyframe_measurements;
    SyncDataReceiver * sync_data_receiver = nullptr;
    bool updated = false;
    std::set<LandmarkIdType> used_landmarks;
    std::recursive_mutex imu_prop_lock;
    
    //Internal functions
    bool tryinitFirstPose(VisualImageDescArray & frame);
    VINSFramePtr addFrame(VisualImageDescArray & _frame);
    VINSFramePtr addFrameRemote(const VisualImageDescArray & _frame);
    void solveNonDistrib();
    void setupImuFactors();
    void setupLandmarkFactors();
    void addIMUFactor(FrameIdType frame_ida, FrameIdType frame_idb, const IntegrationBasePtr& _pre_integration);
    void setupPriorFactor();
    std::pair<bool, Swarm::Pose> initialFramePnP(const VisualImageDescArray & frame, 
        const Swarm::Pose & initial_pose);
    void addSldWinToFrame(VisualImageDescArray & frame);
    void addRemoteImuBuf(int drone_id, const IMUBuffer & imu_buf);
    bool isLocalFrame(FrameIdType frame_id) const;
    bool isMain() const;
    void resetMarginalizer();
    bool hasCommonLandmarkMeasurments();

    //Multi-drone functions
    void onDistributedVinsData(const DistributedVinsData & dist_data);
    void onSyncSignal(int drone_id, int signal, int64_t token);
    void waitForStart();
public:
    std::recursive_mutex frame_mutex;

    D2Estimator(int drone_id);
    void inputImu(IMUData data);
    bool inputImage(VisualImageDescArray & frame);
    void inputRemoteImage(VisualImageDescArray & frame);
    void solveinDistributedMode();
    Swarm::Odometry getImuPropagation();
    Swarm::Odometry getOdometry() const;
    Swarm::Odometry getOdometry(int drone_id) const;
    void init(ros::NodeHandle & nh, D2VINSNet * net);
    D2EstimatorState & getState();
    std::vector<LandmarkPerId> getMarginedLandmarks() const;
    void updateSldwin(int drone_id, const std::vector<FrameIdType> & sld_win);

    //Multi-drone comm protocol
    void sendDistributedVinsData(DistributedVinsData data);
    void sendSyncSignal(SyncSignal data, int64_t token);
    bool readyForStart();
    const std::map<LandmarkIdType, LandmarkPerId> & getLandmarkDB() const;
    const std::vector<VINSFramePtr> & getSelfSldWin() const;
    D2Visualization & getVisualizer();
    void setPGOPoses(const std::map<int, Swarm::Pose> & poses);
    std::set<int> getNearbyDronesbyPGOData(const std::map<int, std::pair<int, Swarm::Pose>> & vins_poses);
    void setStateProperties();
    virtual std::pair<Swarm::Odometry, std::pair<IMUBuffer, int>> getMotionPredict(double stamp) const;
    bool isInitialized() const;
};
}