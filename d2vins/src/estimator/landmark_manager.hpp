#pragma once

#include <d2vins/d2vins_types.hpp>
#include "d2frontend/d2featuretracker.h"

namespace D2VINS {
class D2EstimatorState;
class D2LandmarkManager : public D2FrontEnd::LandmarkManager {
    std::vector<int> frame_ids;
    std::map<FrameIdType, std::set<LandmarkIdType>> related_landmarks;
    std::map<LandmarkIdType, state_type*> landmark_state;
    int estimated_landmark_size = 0;

    void initialLandmarkState(LandmarkPerId & lm, const D2EstimatorState * state);
public:
    virtual void addKeyframe(const VisualImageDescArray & images, double td);
    std::vector<LandmarkPerId> availableMeasurements() const;
    double * getLandmarkState(LandmarkIdType landmark_id) const;
    void initialLandmarks(const D2EstimatorState * state);
    void syncState(const D2EstimatorState * state);
    void popFrame(FrameIdType frame_id);
    std::vector<LandmarkPerId> getInitializedLandmarks() const;
    LandmarkPerId & getLandmark(LandmarkIdType landmark_id);
    FrameIdType getLandmarkBaseFrame(LandmarkIdType landmark_id) const;
    bool hasLandmark(LandmarkIdType landmark_id) const;
    void outlierRejection(const D2EstimatorState * state);
};

}