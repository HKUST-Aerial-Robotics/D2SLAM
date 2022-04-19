#pragma once

#include <d2vins/d2vins_types.hpp>
#include "d2frontend/d2featuretracker.h"

namespace D2VINS {
class D2LandmarkManager : public D2FrontEnd::LandmarkManager {
public:
    std::vector<int> frame_ids;
    std::map<D2FrontEnd::FrameIdType, std::set<D2FrontEnd::LandmarkIdType>> related_landmarks;
    std::map<D2FrontEnd::LandmarkIdType, state_type*> landmark_state;
    virtual void addKeyframe(const D2FrontEnd::VisualImageDescArray & images, double td) {
        for (auto & image : images.images) {
            for (auto lm : image.landmarks) {
                if (lm.landmark_id < 0) {
                    //Do not use unmatched features.
                    continue;
                }
                related_landmarks[images.frame_id].insert(lm.landmark_id);
                lm.cur_td = td;
                updateLandmark(lm);
                if (landmark_state.find(lm.landmark_id) == landmark_state.end()) {
                    if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                        landmark_state[lm.landmark_id] = new state_type[INV_DEP_SIZE];
                    } else {
                        landmark_state[lm.landmark_id] = new state_type[POS_SIZE];
                    }
                }
            }
        }
        printf("[D2VINS::D2LandmarkManager] addKeyframe current kf %ld, landmarks %ld\n", 
            related_landmarks.size(), landmark_state.size());
    }

    std::vector<D2FrontEnd::LandmarkPerId> availableMeasurements() const {
        //Return all avaiable measurements
        std::vector<D2FrontEnd::LandmarkPerId> ret;
        for (auto & it: landmark_db) {
            auto & lm = it.second;
            if (lm.track.size() > params->landmark_estimate_tracks) {
                ret.push_back(lm);
            }
        }
        return ret;
    }
    
    double * getLandmarkState(D2FrontEnd::LandmarkIdType landmark_id) const {
        return landmark_state.at(landmark_id);
    }

    void initialLandmarks(const std::map<D2FrontEnd::FrameIdType, VINSFrame*> & frame_db, const std::vector<Swarm::Pose> & extrinsic) {
        for (auto & it: landmark_db) {
            auto & lm = it.second;
            auto lm_id = it.first;
            auto lm_per_frame = landmark_db.at(lm_id).track[0];
            const auto & firstFrame = *frame_db.at(lm_per_frame.frame_id);
            auto pt2d_n = lm_per_frame.pt2d_norm;
            auto ext = extrinsic[lm_per_frame.camera_id];
            if (lm.track.size() > params->landmark_estimate_tracks) {
                if (lm.flag == D2FrontEnd::LandmarkFlag::UNINITIALIZED) {
                    //Initialize by motion.
                    Vector3d pos(pt2d_n.x(), pt2d_n.y(), 1.0);
                    pos = pos * 10; //Initial to 10 meter away... TODO: Initial with motion
                    pos = firstFrame.odom.pose()*ext*pos;
                    lm.position = pos;
                    lm.flag = D2FrontEnd::LandmarkFlag::INITIALIZED;
                    if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                        *landmark_state[lm_id] = 0.1;
                        printf("[D2VINS::D2LandmarkManager] initialLandmarks (UNINITIALIZED) LM %d inv_dep/dep %.2f/%.2f pos %.2f %.2f %.2f\n",
                            lm_id, *landmark_state[lm_id], 1./(*landmark_state[lm_id]), lm.position.x(), lm.position.y(), lm.position.z());
                    } else {
                        memcpy(landmark_state[lm_id], lm.position.data(), sizeof(state_type)*POS_SIZE);
                    }
                } else if(lm.flag == D2FrontEnd::LandmarkFlag::INITIALIZED) {
                    //Use depth to initial
                    Vector3d pos(pt2d_n.x(), pt2d_n.y(), 1.0);
                    pos = pos* lm_per_frame.depth;
                    pos = firstFrame.odom.pose()*ext*pos;
                    lm.position = pos;
                    if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                        *landmark_state[lm_id] = 1/lm_per_frame.depth;
                        printf("[D2VINS::D2LandmarkManager] initialLandmarks LM %d inv_dep/dep %.2f/%.2f pos %.2f %.2f %.2f\n",
                            lm_id, *landmark_state[lm_id], 1./(*landmark_state[lm_id]), lm.position.x(), lm.position.y(), lm.position.z());
                    } else {
                        memcpy(landmark_state[lm_id], lm.position.data(), sizeof(state_type)*POS_SIZE);
                    }
                } else if(lm.flag == D2FrontEnd::LandmarkFlag::ESTIMATED) {
                    //Extracting depth from estimated pos
                    if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                        Vector3d pos_cam = (firstFrame.odom.pose()*ext).inverse()*lm.position;
                        *landmark_state[lm_id] = 1/pos_cam.z();
                        printf("[D2VINS::D2LandmarkManager] initialLandmarks LM %d inv_dep/dep %.2f/%.2f pos %.2f %.2f %.2f\n",
                            lm_id, *landmark_state[lm_id], 1./(*landmark_state[lm_id]), lm.position.x(), lm.position.y(), lm.position.z());
                    } else {
                        memcpy(landmark_state[lm_id], lm.position.data(), sizeof(state_type)*POS_SIZE);
                    }
                }
            }
        }
    }

    void syncState(const std::vector<Swarm::Pose> & extrinsic, const std::map<D2FrontEnd::FrameIdType, VINSFrame*> & frame_db) {
        //Sync inverse depth to 3D positions
        for (auto it : landmark_state) {
            auto lm_id = it.first;
            auto & lm = landmark_db.at(lm_id);
            if (lm.track.size() > params->landmark_estimate_tracks) {
                if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                    auto inv_dep = *it.second;
                    auto lm_per_frame = lm.track[0];
                    const auto & firstFrame = *frame_db.at(lm_per_frame.frame_id);
                    auto ext = extrinsic[lm_per_frame.camera_id];
                    auto pt2d_n = lm_per_frame.pt2d_norm;
                    Vector3d pos(pt2d_n.x(), pt2d_n.y(), 1.0);
                    pos = pos / inv_dep;
                    pos = firstFrame.odom.pose()*ext*pos;
                    lm.position = pos;
                    lm.flag = D2FrontEnd::LandmarkFlag::ESTIMATED;
                    printf("[D2VINS::D2LandmarkManager] update LM %d inv_dep/dep %.2f/%.2f pos %.2f %.2f %.2f\n",
                        lm_id, inv_dep, 1./inv_dep, pos.x(), pos.y(), pos.z());
                } else {
                    lm.position.x() = it.second[0];
                    lm.position.y() = it.second[1];
                    lm.position.z() = it.second[2];
                }
            }
        }
    }
    
    void popFrame(D2FrontEnd::FrameIdType frame_id) {
        if (related_landmarks.find(frame_id) == related_landmarks.end()) {
            return;
        }
        auto _landmark_ids = related_landmarks[frame_id];
        for (auto _id : _landmark_ids) {
            auto & lm = landmark_db.at(_id);
            auto _size = lm.popFrame(frame_id);
            if (_size == 0) {
                //Remove this landmark.
                landmark_db.erase(_id);
                landmark_state.erase(_id);
            }
        }
        related_landmarks.erase(frame_id);
    }

};

}