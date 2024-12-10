#include <d2frontend/d2frontend_params.h>
#include <d2frontend/d2landmark_manager.h>

namespace D2FrontEnd {

int LandmarkManager::addLandmark(const LandmarkPerFrame &lm) {
  const Guard lock(state_lock);
  auto _id = count + MAX_FEATURE_NUM * params->self_id;
  count++;
  LandmarkPerFrame lm_copy = lm;
  lm_copy.landmark_id = _id;
  landmark_db[_id] = lm_copy;
  related_landmarks[lm_copy.frame_id][_id] =
      related_landmarks[lm_copy.frame_id][_id] + 1;
  total_lm_per_frame_num++;
  return _id;
}

void LandmarkManager::updateLandmark(const LandmarkPerFrame &lm) {
  const Guard lock(state_lock);
  if (lm.landmark_id < 0) {
    return;
  }
  if (landmark_db.find(lm.landmark_id) == landmark_db.end()) {
    landmark_db[lm.landmark_id] = lm;
  } else {
    landmark_db.at(lm.landmark_id).add(lm);
  }
  total_lm_per_frame_num++;
  related_landmarks[lm.frame_id][lm.landmark_id] =
      related_landmarks[lm.frame_id][lm.landmark_id] + 1;
}

void LandmarkManager::removeLandmark(const LandmarkIdType &id) {
  const Guard lock(state_lock);
  landmark_db.erase(id);
}

std::vector<LandmarkPerId> LandmarkManager::popFrame(FrameIdType frame_id,
                                                     bool pop_base) {
  const Guard lock(state_lock);
  // Returning margined landmarks
  std::vector<LandmarkPerId> margined_landmarks;
  if (related_landmarks.find(frame_id) == related_landmarks.end()) {
    return margined_landmarks;
  }
  auto _landmark_ids = related_landmarks[frame_id];
  for (auto it : _landmark_ids) {
    auto _id = it.first;
    if (landmark_db.find(_id) == landmark_db.end()) {
      continue;
    }
    auto &lm = landmark_db.at(_id);
    if (pop_base) {
      if (related_landmarks.find(lm.base_frame_id) != related_landmarks.end()) {
        related_landmarks.at(lm.base_frame_id).erase(lm.landmark_id);
      }
      if (lm.base_frame_id != frame_id) {
        // printf("[D2VINS::D2LandmarkManager] popFrame base_frame_id %ld !=
        // frame_id %ld\n", lm.base_frame_id, frame_id);
      }
      lm.popBaseFrame();
    }
    auto _size = lm.popFrame(frame_id);
    total_lm_per_frame_num--;
    if (_size == 0) {
      // Remove this landmark.
      margined_landmarks.emplace_back(lm);
      removeLandmark(_id);
    }
  }
  related_landmarks.erase(frame_id);
  return margined_landmarks;
}

std::vector<LandmarkPerId> LandmarkManager::getInitializedLandmarks(
    int min_tracks) const {
  const Guard lock(state_lock);
  std::vector<LandmarkPerId> lm_per_frame_vec;
  for (const auto &it: landmark_db) {
    const auto &lm = it.second;
    if (lm.track.size() >= min_tracks && lm.flag >= LandmarkFlag::INITIALIZED) {
      lm_per_frame_vec.push_back(lm);
    }
  }
  return lm_per_frame_vec;
}

bool LandmarkManager::hasLandmark(LandmarkIdType landmark_id) const {
  const Guard lock(state_lock);
  return landmark_db.find(landmark_id) != landmark_db.end();
}

FrameIdType LandmarkManager::getLandmarkBaseFrame(
    LandmarkIdType landmark_id) const {
  const Guard lock(state_lock);
  return landmark_db.at(landmark_id).track[0].frame_id;
}

std::vector<LandmarkIdType> LandmarkManager::findCommonLandmarkIds(
    FrameIdType frame_id1, FrameIdType frame_id2) const {
  const Guard lock(state_lock);
  auto lm_last = getRelatedLandmarks(frame_id1);
  auto lm_second_last = getRelatedLandmarks(frame_id2);
  std::set<LandmarkIdType> common_lm;
  std::set_intersection(lm_last.begin(), lm_last.end(), lm_second_last.begin(),
                        lm_second_last.end(),
                        std::inserter(common_lm, common_lm.begin()));
  return std::vector<LandmarkIdType>(common_lm.begin(), common_lm.end());
}

std::vector<std::pair<LandmarkPerFrame, LandmarkPerFrame>>
LandmarkManager::findCommonLandmarkPerFrames(FrameIdType frame_id1,
                                             FrameIdType frame_id2) const {
  const Guard lock(state_lock);
  auto lm_common = findCommonLandmarkIds(frame_id1, frame_id2);
  std::vector<std::pair<LandmarkPerFrame, LandmarkPerFrame>> ret;
  for (auto lm_id : lm_common) {
    auto &lm = landmark_db.at(lm_id);
    LandmarkPerFrame lm1 = lm.at(frame_id1);
    LandmarkPerFrame lm2 = lm.at(frame_id2);
    ret.emplace_back(lm1, lm2);
  }
  return ret;
}

}  // namespace D2FrontEnd