#pragma once
#include "landmark_manager.hpp"
#include <d2common/d2state.hpp>
#include <d2common/d2vinsframe.h>

using namespace Eigen;
using namespace D2Common;

namespace D2VINS {
class Marginalizer;
class PriorFactor;
class D2EstimatorState : public D2State {
protected:
    std::map<int, std::vector<VINSFrame*>> sld_wins;
    std::map<int, std::vector<FrameIdType>> latest_remote_sld_wins;
    std::map<FrameIdType, int> frame_indices;
    D2LandmarkManager lmanager;
    std::map<FrameIdType, state_type*> _frame_spd_Bias_state;
    std::map<CamIdType, state_type*> _camera_extrinsic_state;
    std::map<CamIdType, int> camera_drone;
    std::map<CamIdType, Swarm::Pose> extrinsic; //extrinsic of cameras by ID

    std::vector<LandmarkPerId> popFrame(int index);
    std::vector<LandmarkPerId> removeFrameById(FrameIdType frame_id, bool remove_base=false); 
        //If remove base, will remove the relevant landmarks' base frame.
        //This is for marginal the keyframes that not is baseframe of all landmarks (in multi-drone)
    void outlierRejection();
    void updateSldWinsIMU(const std::map<int, IMUBuffer> & remote_imu_bufs);
    Marginalizer * marginalizer = nullptr;
    PriorFactor * prior_factor = nullptr;
    bool marginalized_self_first = false;

public:
    state_type td = 0.0;
    D2EstimatorState(int _self_id);

    void init(std::vector<Swarm::Pose> _extrinsic, double _td);

    //Get states
    int getPoseIndex(FrameIdType frame_id) const;
    double * getExtrinsicState(int i) const;
    double * getSpdBiasState(FrameIdType frame_id) const;
    double * getLandmarkState(LandmarkIdType landmark_id) const;
    double * getTdState(int drone_id);
    double getTd(int drone_id);
    PriorFactor * getPrior() const;
    FrameIdType getLandmarkBaseFrame(LandmarkIdType landmark_id) const;
    Swarm::Pose getExtrinsic(CamIdType cam_id) const;
    std::set<CamIdType> getAvailableCameraIds() const;
    std::vector<LandmarkPerId> availableLandmarkMeasurements() const;
    std::vector<LandmarkPerId> getInitializedLandmarks() const;
    std::vector<LandmarkPerId> getRelatedLandmarks(FrameIdType frame_id) const;
    LandmarkPerId & getLandmarkbyId(LandmarkIdType id);
    bool hasLandmark(LandmarkIdType id) const;
    int getCameraBelonging(CamIdType cam_id) const;

    //Frame operations
    std::vector<LandmarkPerId> clearFrame();
    void addFrame(const VisualImageDescArray & images, const VINSFrame & _frame);
    void addCamera(const Swarm::Pose & pose, int camera_index, int drone_id, CamIdType camera_id=-1);
    bool hasCamera(CamIdType frame_id) const;
    void updateSldwin(int drone_id, const std::vector<FrameIdType> & sld_win);
    virtual void moveAllPoses(int new_ref_frame_id, const Swarm::Pose & delta_pose) override;
    const std::vector<VINSFrame*> & getSldWin(int drone_id) const;

    //Frame access    
    VINSFrame & getFrame(int index);
    const VINSFrame & getFrame(int index) const;
    VINSFrame & firstFrame();
    const VINSFrame & lastFrame() const;
    VINSFrame & lastFrame();
    size_t size() const;
    VINSFrame & getFrame(int drone_id, int index);
    Swarm::Pose getEstimatedPose(int drone_id, int index) const;
    Swarm::Pose getEstimatedPose(FrameIdType frame_id) const;
    Swarm::Odometry getEstimatedOdom(FrameIdType frame_id) const;
    const VINSFrame & getFrame(int drone_id, int index) const;
    VINSFrame & firstFrame(int drone_id);
    const VINSFrame &  lastFrame(int drone_id) const;
    VINSFrame & lastFrame(int drone_id);
    size_t size(int drone_id) const;

    //Solving process
    void syncFromState();
    void preSolve(const std::map<int, IMUBuffer> & remote_imu_bufs);
    void repropagateIMU();

    //Debug
    void printSldWin(const std::map<FrameIdType, int> & keyframe_measurments) const;
    bool marginalizeSelf() const {
        return marginalized_self_first;
    }

    void setMarginalizer(Marginalizer * _marginalizer) {
        marginalizer = _marginalizer;
    }
    const std::map<LandmarkIdType, LandmarkPerId> & getLandmarkDB() const {
        return lmanager.getLandmarkDB();
    }
};
}