#pragma once

#include <d2common/d2vinsframe.h>
#include "d2frontend/d2landmark_manager.h"

namespace D2VINS {
class D2EstimatorState;
class D2LandmarkManager : public D2FrontEnd::LandmarkManager {
    std::map<LandmarkIdType, state_type*> landmark_state;
    int estimated_landmark_size = 0;
    void initialLandmarkState(LandmarkPerId & lm, const D2EstimatorState * state);
public:
    virtual void addKeyframe(const VisualImageDescArray & images, double td);
    std::vector<LandmarkPerId> availableMeasurements(int max_pts) const;
    double * getLandmarkState(LandmarkIdType landmark_id) const;
    void initialLandmarks(const D2EstimatorState * state);
    void syncState(const D2EstimatorState * state);
    void outlierRejection(const D2EstimatorState * state, const std::set<LandmarkIdType> & used_landmarks);
    void moveByPose(const Swarm::Pose & delta_pose);
    virtual void removeLandmark(const LandmarkIdType & id) override;
};

}