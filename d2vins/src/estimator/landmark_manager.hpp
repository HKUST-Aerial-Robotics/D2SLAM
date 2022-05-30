#pragma once

#include <d2common/d2vinsframe.h>
#include "d2frontend/d2landmark_manager.h"

namespace D2VINS {
class D2EstimatorState;
class D2LandmarkManager : public D2FrontEnd::LandmarkManager {
    std::vector<int> frame_ids;
    std::map<FrameIdType, std::set<LandmarkIdType>> related_landmarks;
    std::map<LandmarkIdType, state_type*> landmark_state;
    int estimated_landmark_size = 0;

    void initialLandmarkState(LandmarkPerId & lm, const D2EstimatorState * state);
    typedef std::lock_guard<std::recursive_mutex> Guard;
    mutable std::recursive_mutex state_lock;
public:
    virtual void addKeyframe(const VisualImageDescArray & images, double td);
    std::vector<LandmarkPerId> availableMeasurements() const;
    double * getLandmarkState(LandmarkIdType landmark_id) const;
    void initialLandmarks(const D2EstimatorState * state);
    void syncState(const D2EstimatorState * state);
    std::vector<LandmarkPerId> popFrame(FrameIdType frame_id, bool pop_base=false); //If pop base, we will remove the related landmarks' base frame.
    std::vector<LandmarkPerId> getInitializedLandmarks() const;
    LandmarkPerId & getLandmark(LandmarkIdType landmark_id);
    FrameIdType getLandmarkBaseFrame(LandmarkIdType landmark_id) const;
    bool hasLandmark(LandmarkIdType landmark_id) const;
    void outlierRejection(const D2EstimatorState * state);
    std::vector<LandmarkPerId> getRelatedLandmarks(FrameIdType frame_id) const;
};

}