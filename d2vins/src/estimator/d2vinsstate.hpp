#pragma once
#include "landmark_manager.hpp"
#include <d2vins/d2vins_types.hpp>

using namespace Eigen;
namespace D2VINS {
class Marginalizer;
class PriorFactor;
class D2EstimatorState {
protected:
    std::vector<VINSFrame*> sld_win;
    std::map<FrameIdType, VINSFrame*> frame_db;
    std::map<FrameIdType, int> frame_indices;
    D2LandmarkManager lmanager;
    std::map<FrameIdType, state_type*> _frame_pose_state;
    std::map<FrameIdType, state_type*> _frame_spd_Bias_state;
    std::map<CamIdType, state_type*> _camera_extrinsic_state;
    std::map<CamIdType, Swarm::Pose> extrinsic; //extrinsic of cameras by ID

    std::vector<LandmarkPerId> popFrame(int index);
    void outlierRejection();
    void updatePoseIndices();
    Marginalizer * marginalizer = nullptr;
    PriorFactor * prior_factor = nullptr;
    
public:
    state_type td = 0.0;
    D2EstimatorState() {}
    void init(std::vector<Swarm::Pose> _extrinsic, double _td);
    size_t size() const;

    //Get states
    double * getPoseState(FrameIdType frame_id) const;
    int getPoseIndex(FrameIdType frame_id) const;
    double * getExtrinsicState(int i) const;
    double * getSpdBiasState(FrameIdType frame_id) const;
    double * getLandmarkState(LandmarkIdType landmark_id) const;
    double * getTdState(int camera_index);
    PriorFactor * getPrior() const;
    FrameIdType getLandmarkBaseFrame(LandmarkIdType landmark_id) const;
    Swarm::Pose getExtrinsic(CamIdType cam_id) const;
    std::set<CamIdType> getAvailableCameraIds() const;
    std::vector<LandmarkPerId> availableLandmarkMeasurements() const;
    std::vector<LandmarkPerId> getInitializedLandmarks() const;
    LandmarkPerId & getLandmarkbyId(LandmarkIdType id);
    bool hasLandmark(LandmarkIdType id) const;

    //Frame operations
    std::vector<LandmarkPerId> clearFrame();
    void addFrame(const VisualImageDescArray & images, const VINSFrame & _frame, bool is_keyframe);

    //Frame access    
    VINSFrame & getFrame(int index);
    const VINSFrame & getFramebyId(int frame_id) const;
    VINSFrame & firstFrame();
    VINSFrame lastFrame() const;
    
    //Solving process
    void syncFromState();
    void preSolve();

    //Debug
    void printSldWin() const;

    void setMarginalizer(Marginalizer * _marginalizer) {
        marginalizer = _marginalizer;
    }
};
}