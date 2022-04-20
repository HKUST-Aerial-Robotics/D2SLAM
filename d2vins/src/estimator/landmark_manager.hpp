#pragma once

#include <d2vins/d2vins_types.hpp>
#include "d2frontend/d2featuretracker.h"

namespace D2VINS {
class D2LandmarkManager : public D2FrontEnd::LandmarkManager {
    std::vector<int> frame_ids;
    std::map<D2FrontEnd::FrameIdType, std::set<D2FrontEnd::LandmarkIdType>> related_landmarks;
    std::map<D2FrontEnd::LandmarkIdType, state_type*> landmark_state;
public:
    virtual void addKeyframe(const D2FrontEnd::VisualImageDescArray & images, double td);
    std::vector<D2FrontEnd::LandmarkPerId> availableMeasurements() const;
    double * getLandmarkState(D2FrontEnd::LandmarkIdType landmark_id) const;
    void initialLandmarks(const std::map<D2FrontEnd::FrameIdType, VINSFrame*> & frame_db, const std::vector<Swarm::Pose> & extrinsic);
    void syncState(const std::vector<Swarm::Pose> & extrinsic, const std::map<D2FrontEnd::FrameIdType, VINSFrame*> & frame_db);
    void popFrame(D2FrontEnd::FrameIdType frame_id);
    std::vector<D2FrontEnd::LandmarkPerId> getInitializedLandmarks() const;

};

}