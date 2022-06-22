#pragma once
#include "d2common/d2landmarks.h"

using namespace D2Common;
#define MAX_FEATURE_NUM 10000000

namespace D2FrontEnd {
class LandmarkManager {
protected:
    std::map<FrameIdType, std::set<LandmarkIdType>> related_landmarks;
    std::map<LandmarkIdType, LandmarkPerId> landmark_db;
    int count = 0;
    typedef std::lock_guard<std::recursive_mutex> Guard;
    mutable std::recursive_mutex state_lock;
public:
    int total_lm_per_frame_num = 0;
    virtual int addLandmark(const LandmarkPerFrame & lm);
    virtual void updateLandmark(const LandmarkPerFrame & lm);
    bool hasLandmark(const LandmarkIdType & id) const {
        return landmark_db.find(id) != landmark_db.end();
    }
    LandmarkPerId & at(LandmarkIdType i) {
        return landmark_db.at(i);
    }
    std::vector<LandmarkPerId> popFrame(FrameIdType frame_id, bool pop_base=false); //If pop base, we will remove the related landmarks' base frame.
    virtual void removeLandmark(const LandmarkIdType & id);
    const std::map<LandmarkIdType, LandmarkPerId> & getLandmarkDB() const {
        return landmark_db;
    }
};
}