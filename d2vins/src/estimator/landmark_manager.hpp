#pragma once

#include <d2vins/d2vins_types.hpp>
#include "d2frontend/d2featuretracker.h"

namespace D2VINS {
class D2LandmarkManager : public D2FrontEnd::LandmarkManager {
    std::vector<int> frame_ids;
    std::map<FrameIdType, std::set<LandmarkIdType>> related_landmarks;
    std::map<LandmarkIdType, state_type*> landmark_state;
    int estimated_landmark_size = 0;
public:
    virtual void addKeyframe(const VisualImageDescArray & images, double td);
    std::vector<LandmarkPerId> availableMeasurements() const;
    double * getLandmarkState(LandmarkIdType landmark_id) const;
    void initialLandmarks(const std::map<FrameIdType, VINSFrame*> & frame_db, const std::vector<Swarm::Pose> & extrinsic);
    void syncState(const std::vector<Swarm::Pose> & extrinsic, const std::map<FrameIdType, VINSFrame*> & frame_db);
    void popFrame(FrameIdType frame_id);
    std::vector<LandmarkPerId> getInitializedLandmarks() const;
    LandmarkPerId & getLandmark(LandmarkIdType landmark_id);
    bool hasLandmark(LandmarkIdType landmark_id) const;
    void outlierRejection(const std::map<FrameIdType, VINSFrame*> & frame_db, const std::vector<Swarm::Pose> & extrinsic);
};

}