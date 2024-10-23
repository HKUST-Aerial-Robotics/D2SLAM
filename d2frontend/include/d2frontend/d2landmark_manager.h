#pragma once
#include "d2common/d2landmarks.h"

using namespace D2Common;
#define MAX_FEATURE_NUM 10000000

namespace D2FrontEnd {
class LandmarkManager {
protected:
    std::map<FrameIdType, std::map<LandmarkIdType, int>> related_landmarks;
    std::map<LandmarkIdType, LandmarkPerId> landmark_db;
    int count = 0;
    typedef std::lock_guard<std::recursive_mutex> Guard;
    mutable std::recursive_mutex state_lock;

public:
    int total_lm_per_frame_num = 0;
    virtual int addLandmark(const LandmarkPerFrame & lm);
    virtual void updateLandmark(const LandmarkPerFrame & lm);
    LandmarkPerId & at(LandmarkIdType i) {
        if (landmark_db.find(i) == landmark_db.end()) {
            // Throw error with i
            std::cout << "Landmark id " << i << " not found!" << std::endl;
            throw std::runtime_error("Landmark id not found!");
        }
        return landmark_db.at(i);
    }
    const LandmarkPerId & at(LandmarkIdType i) const {
        return landmark_db.at(i);
    }
    std::vector<LandmarkPerId> popFrame(FrameIdType frame_id, bool pop_base=false); //If pop base, we will remove the related landmarks' base frame.
    virtual void removeLandmark(const LandmarkIdType & id);
    const std::map<LandmarkIdType, LandmarkPerId> & getLandmarkDB() const {
        return landmark_db;
    }
    std::set<LandmarkIdType> getRelatedLandmarks(FrameIdType frame_id) const {
        Guard g(state_lock);
        if (related_landmarks.find(frame_id) == related_landmarks.end()) {
            return std::set<LandmarkIdType>();
        }
        std::set<LandmarkIdType> lms;
        for (auto lm : related_landmarks.at(frame_id)) {
            if (lm.second > 0) {
                lms.insert(lm.first);
            }
        }
        return lms;
    }
    std::vector<LandmarkPerId> getInitializedLandmarks(int min_tracks) const;
    FrameIdType getLandmarkBaseFrame(LandmarkIdType landmark_id) const;
    bool hasLandmark(LandmarkIdType landmark_id) const;
    std::vector<LandmarkIdType> findCommonLandmarkIds(FrameIdType frame_id1, FrameIdType frame_id2) const;
    std::vector<std::pair<LandmarkPerFrame,LandmarkPerFrame>> findCommonLandmarkPerFrames(FrameIdType frame_id1, FrameIdType frame_id2) const;

};
}