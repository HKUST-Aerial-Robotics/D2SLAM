#include <d2frontend/d2landmark_manager.h>
#include <d2frontend/d2frontend_params.h>

namespace D2FrontEnd {
    
int LandmarkManager::addLandmark(const LandmarkPerFrame & lm) {
    auto _id = count + MAX_FEATURE_NUM*params->self_id;
    count ++;
    LandmarkPerFrame lm_copy = lm;
    lm_copy.landmark_id = _id;
    landmark_db[_id] = lm_copy;
    related_landmarks[lm_copy.frame_id].insert(_id);
    total_lm_per_frame_num ++;
    return _id;
}

void LandmarkManager::updateLandmark(const LandmarkPerFrame & lm) {
    if (lm.landmark_id < 0) {
        return;
    }
    if (landmark_db.find(lm.landmark_id) == landmark_db.end()) {
        landmark_db[lm.landmark_id] = lm;
    } else {
        landmark_db.at(lm.landmark_id).add(lm);
    }
    total_lm_per_frame_num ++;
    related_landmarks[lm.frame_id].insert(lm.landmark_id);
    assert(lm.landmark_id >= 0 && "landmark id must > 0");
}

void LandmarkManager::removeLandmark(const LandmarkIdType & id) {
    landmark_db.erase(id);
}

std::vector<LandmarkPerId> LandmarkManager::popFrame(FrameIdType frame_id, bool pop_base) {
    const Guard lock(state_lock);
    //Returning margined landmarks
    std::vector<LandmarkPerId> margined_landmarks;
    if (related_landmarks.find(frame_id) == related_landmarks.end()) {
        return margined_landmarks;
    }
    auto _landmark_ids = related_landmarks[frame_id];
    for (auto _id : _landmark_ids) {
        auto & lm = landmark_db.at(_id);
        if (pop_base) {
            if (related_landmarks.find(lm.base_frame_id)!=related_landmarks.end()) {
                related_landmarks.at(lm.base_frame_id).erase(lm.landmark_id);
            }
            if (lm.base_frame_id != frame_id) {
                // printf("[D2VINS::D2LandmarkManager] popFrame base_frame_id %ld != frame_id %ld\n", lm.base_frame_id, frame_id);
            }
            lm.popBaseFrame();
        }
        auto _size = lm.popFrame(frame_id);
        total_lm_per_frame_num --;
        if (_size == 0) {
            //Remove this landmark.
            margined_landmarks.emplace_back(lm);
            removeLandmark(_id);
        }
    }
    related_landmarks.erase(frame_id);
    return margined_landmarks;
}


}