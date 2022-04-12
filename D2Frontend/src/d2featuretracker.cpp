#include <d2frontend/d2featuretracker.h>

namespace D2Frontend {

bool D2FeatureTracker::track(VisualImageDescArray * frames) {
    if (current_keyframe == nullptr) {
        current_keyframe = frames;
    } else {
        
    }
}

bool D2FeatureTracker::track(VisualImageDesc & frame) {
    auto & previous = current_keyframe->images[frame.camera_id];
    auto prev_pts = previous.landmarks_2d;
    auto cur_pts = previous.landmarks_2d;
    std::vector<int> ids_up;
    std::vector<int> ids_down;
    match_local_features(prev_pts, cur_pts, previous.feature_descriptor, 
        frame.feature_descriptor, ids_up, ids_down);
}


}