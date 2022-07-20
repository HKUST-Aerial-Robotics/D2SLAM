#include "landmark_manager.hpp"
#include "d2vinsstate.hpp"
#include "../d2vins_params.hpp"

namespace D2VINS {

double triangulatePoint3DPts(std::vector<Swarm::Pose> poses, std::vector<Vector3d> &points, Vector3d &point_3d);

void D2LandmarkManager::addKeyframe(const VisualImageDescArray & images, double td) {
    const Guard lock(state_lock);
    for (auto & image : images.images) {
        for (auto lm : image.landmarks) {
            if (lm.landmark_id < 0) {
                //Do not use unmatched features.
                continue;
            }
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
}

std::vector<LandmarkPerId> D2LandmarkManager::availableMeasurements() const {
    //Return all avaiable measurements
    const Guard lock(state_lock);
    std::vector<LandmarkPerId> ret;
    for (auto & it: landmark_db) {
        auto & lm = it.second;
        if (lm.track.size() >= params->landmark_estimate_tracks && 
            lm.flag >= LandmarkFlag::INITIALIZED && lm.flag != LandmarkFlag::OUTLIER) {
            ret.push_back(lm);
        }
    }
    return ret;
}

double * D2LandmarkManager::getLandmarkState(LandmarkIdType landmark_id) const {
    const Guard lock(state_lock);
    return landmark_state.at(landmark_id);
}

FrameIdType D2LandmarkManager::getLandmarkBaseFrame(LandmarkIdType landmark_id) const {
    const Guard lock(state_lock);
    return landmark_db.at(landmark_id).track[0].frame_id;
}

void D2LandmarkManager::moveByPose(const Swarm::Pose & delta_pose) {
    const Guard lock(state_lock);
    for (auto it: landmark_db) {
        auto & lm = it.second;
        if (lm.flag != LandmarkFlag::UNINITIALIZED) {
            lm.position = delta_pose * lm.position;
        }
    }
}

void D2LandmarkManager::initialLandmarkState(LandmarkPerId & lm, const D2EstimatorState * state) {
    const Guard lock(state_lock);
    LandmarkPerFrame lm_first;
    lm_first = lm.track[0];
    auto lm_id = lm.landmark_id;
    auto pt3d_n = lm_first.pt3d_norm;
    auto firstFrame = *state->getFramebyId(lm_first.frame_id);
    // printf("[D2VINS::D2LandmarkManager] Try initial landmark %ld dep %d tracks %ld\n", lm_id, 
    //     lm.track[0].depth_mea && lm.track[0].depth > params->min_depth_to_fuse && lm.track[0].depth < params->max_depth_to_fuse,
    //     lm.track.size());
    if (lm_first.depth_mea && lm_first.depth > params->min_depth_to_fuse && lm_first.depth < params->max_depth_to_fuse) {
        //Use depth to initial
        auto ext = state->getExtrinsic(lm_first.camera_id);
        //Note in depth mode, pt3d = (u, v, w), depth is distance since we use unitsphere
        Vector3d pos = pt3d_n * lm_first.depth;
        pos = firstFrame.odom.pose()*ext*pos;
        lm.position = pos;
        if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
            *landmark_state[lm_id] = 1/lm_first.depth;
            if (params->debug_print_states) {
                printf("[D2VINS::D2LandmarkManager] Initialize landmark %ld by depth measurement position %.3f %.3f %.3f inv_dep %.3f\n",
                    lm_id, pos.x(), pos.y(), pos.z(), 1/lm_first.depth);
            }
        } else {
            memcpy(landmark_state[lm_id], lm.position.data(), sizeof(state_type)*POS_SIZE);
        }
        lm.flag = LandmarkFlag::INITIALIZED;
    } else if (lm.track.size() >= params->landmark_estimate_tracks) {
        //Initialize by motion.
        std::vector<Swarm::Pose> poses;
        std::vector<Vector3d> points;
        auto ext_base = state->getExtrinsic(lm_first.camera_id);
        Eigen::Vector3d _min = (firstFrame.odom.pose()*ext_base).pos();
        Eigen::Vector3d _max = (firstFrame.odom.pose()*ext_base).pos();
        for (auto & it: lm.track) {
            // printf("[D2VINS::D2LandmarkManager] Initialize landmark %ld by motion frame %d camera_id %d index %d\n", 
                // lm_id, it.frame_id, it.camera_id, it.camera_index);
            auto frame = *state->getFramebyId(it.frame_id);
            auto ext = state->getExtrinsic(it.camera_id);
            poses.push_back(frame.odom.pose()*ext);
            points.push_back(it.pt3d_norm);
            _min = _min.cwiseMin((frame.odom.pose()*ext).pos());
            _max = _max.cwiseMax((frame.odom.pose()*ext).pos());
        }
        if ((_max - _min).norm() > params->depth_estimate_baseline) {
            //Initialize by triangulation
            Vector3d point_3d(0., 0., 0.);
            if (triangulatePoint3DPts(poses, points, point_3d) < params->tri_max_err) {
                lm.position = point_3d;
                if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                    auto ptcam = (firstFrame.odom.pose()*ext_base).inverse()*point_3d;
                    auto inv_dep = 1/ptcam.norm();
                    if (inv_dep > params->min_inv_dep) {
                        lm.flag = LandmarkFlag::INITIALIZED;
                        *landmark_state[lm_id] = inv_dep;
                        if (params->debug_print_states) {
                            printf("[D2VINS::D2LandmarkManager] Initialize landmark %ld tracks %ld baseline %.2f by triangulation position %.3f %.3f %.3f inv_dep %.3f\n",
                                lm_id, lm.track.size(), (_max - _min).norm(), point_3d.x(), point_3d.y(), point_3d.z(), inv_dep);
                        }
                    } else {
                        lm.flag = LandmarkFlag::INITIALIZED;
                        *landmark_state[lm_id] = params->min_inv_dep;
                        if (params->debug_print_states) {
                            printf("\033[0;31m [D2VINS::D2LandmarkManager] Initialize failed too far away: landmark %ld tracks %ld baseline %.2f by triangulation position %.3f %.3f %.3f inv_dep %.3f \033[0m\n",
                                lm_id, lm.track.size(), (_max - _min).norm(), point_3d.x(), point_3d.y(), point_3d.z(), inv_dep);
                        }
                    }
                } else {
                    lm.flag = LandmarkFlag::INITIALIZED;
                    memcpy(landmark_state[lm_id], lm.position.data(), sizeof(state_type)*POS_SIZE);
                }
            } else {
                if (params->debug_print_states) {
                    printf("\033[0;31m [D2VINS::D2LandmarkManager] Initialize failed too large triangle error: landmark %ld tracks %ld baseline %.2f by triangulation position %.3f %.3f %.3f\033[0m\n",
                        lm_id, lm.track.size(), (_max - _min).norm(), point_3d.x(), point_3d.y(), point_3d.z());
                }
            }
        } else  { 
            if (params->debug_print_states) {
                printf("\033[0;31m [D2VINS::D2LandmarkManager] Initialize failed too short baseline: landmark %ld tracks %ld baseline %.2f\033[0m\n",
                    lm_id, lm.track.size(), (_max - _min).norm());
            }
        }
    }
}

void D2LandmarkManager::initialLandmarks(const D2EstimatorState * state) {
    const Guard lock(state_lock);
    int inited_count = 0;
    for (auto & it: landmark_db) {
        auto & lm = it.second;
        auto lm_id = it.first;
        //Set to unsolved
        lm.solver_flag = LandmarkSolverFlag::UNSOLVED;
        if (lm.flag == LandmarkFlag::UNINITIALIZED) {
            if (lm.track.size() == 0) {
                printf("\033[0;31m[D2VINS::D2LandmarkManager] Initialize landmark %ld failed, no track.\033[0m\n", lm_id);
                continue;
            }
            initialLandmarkState(lm, state);
            inited_count += 1;
        } else if(lm.flag == LandmarkFlag::ESTIMATED) {
            //Extracting depth from estimated pos
            inited_count += 1;
            if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                auto lm_per_frame = landmark_db.at(lm_id).track[0];
                auto firstFrame = state->getFramebyId(lm_per_frame.frame_id);
                auto ext = state->getExtrinsic(lm_per_frame.camera_id);
                Vector3d pos_cam = (firstFrame->odom.pose()*ext).inverse()*lm.position;
                *landmark_state[lm_id] = 1.0/pos_cam.norm();
            } else {
                memcpy(landmark_state[lm_id], lm.position.data(), sizeof(state_type)*POS_SIZE);
            }
        }
    }

    if (params->debug_print_states) {
        printf("[D2VINS::D2LandmarkManager] Total %d initialized %d avail %d landmarks\n", 
            landmark_db.size(), inited_count, availableMeasurements().size());
    }
}

void D2LandmarkManager::outlierRejection(const D2EstimatorState * state) {
    const Guard lock(state_lock);
    int remove_count = 0;
    int total_count = 0;
    if (estimated_landmark_size < params->perform_outlier_rejection_num) {
        return;
    }
    for (auto & it: landmark_db) {
        auto & lm = it.second;
        auto lm_id = it.first;
        if(lm.flag == LandmarkFlag::ESTIMATED) {
            double err_sum = 0;
            double err_cnt = 0;
            total_count ++;
            for (int i = 1; i < lm.track.size(); i ++) {
                auto pose = state->getFramebyId(lm.track[i].frame_id)->odom.pose();
                auto ext = state->getExtrinsic(lm.track[i].camera_id);
                auto pt3d_n = lm.track[i].pt3d_norm;
                Vector3d pos_cam = (pose*ext).inverse()*lm.position;
                pos_cam.normalize();
                //Compute reprojection error
                Vector3d reproj_error = pt3d_n - pos_cam;
                // printf("[D2VINS::D2LandmarkManager] outlierRejection LM %d inv_dep/dep %.2f/%.2f pos %.2f %.2f %.2f reproj_error %.2f %.2f\n",
                    // lm_id, *landmark_state[lm_id], 1./(*landmark_state[lm_id]), lm.position.x(), lm.position.y(), lm.position.z(), reproj_error.x(), reproj_error.y());
                err_sum += reproj_error.norm();
                err_cnt += 1;
            }
            if (err_cnt > 0) {
                double reproj_err = err_sum/err_cnt;
                if (reproj_err*params->focal_length > params->landmark_outlier_threshold) {
                    remove_count ++;
                    lm.flag = LandmarkFlag::OUTLIER;
                    // printf("[D2VINS::D2LandmarkManager] outlierRejection LM %d inv_dep/dep %.2f/%.2f pos %.2f %.2f %.2f reproj_error %.2f\n",
                    //     lm_id, *landmark_state[lm_id], 1./(*landmark_state[lm_id]), lm.position.x(), lm.position.y(), lm.position.z(), reproj_err*params->focal_length);
                }
            }
        }
    }
    printf("[D2VINS::D2LandmarkManager] outlierRejection remove %d/%d landmarks\n", remove_count, total_count);
}

void D2LandmarkManager::syncState(const D2EstimatorState * state) {
    const Guard lock(state_lock);
    //Sync inverse depth to 3D positions
    estimated_landmark_size = 0;
    for (auto it : landmark_state) {
        auto lm_id = it.first;
        auto & lm = landmark_db.at(lm_id);
        if (lm.solver_flag == LandmarkSolverFlag::SOLVED) {
            if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                auto inv_dep = *it.second;
                auto lm_per_frame = lm.track[0];
                const auto & firstFrame = state->getFramebyId(lm_per_frame.frame_id);
                auto ext = state->getExtrinsic(lm_per_frame.camera_id);
                auto pt3d_n = lm_per_frame.pt3d_norm;
                Vector3d pos = pt3d_n / inv_dep;
                pos = firstFrame->odom.pose()*ext*pos;
                lm.position = pos;
                lm.flag = LandmarkFlag::ESTIMATED;
                if (params->debug_print_states) {
                    printf("[D2VINS::D2LandmarkManager] update LM %d inv_dep/dep %.2f/%.2f depmea %d %.2f pt3d_n %.2f %.2f %.2f pos %.2f %.2f %.2f baseFrame %ld pose %s extrinsic %s\n",
                        lm_id, inv_dep, 1./inv_dep, lm_per_frame.depth_mea, lm_per_frame.depth, 
                            pt3d_n.x(), pt3d_n.y(), pt3d_n.z(),
                            pos.x(), pos.y(), pos.z(),
                            lm_per_frame.frame_id, firstFrame->odom.pose().toStr().c_str(), ext.toStr().c_str());
                }
            } else {
                lm.position.x() = it.second[0];
                lm.position.y() = it.second[1];
                lm.position.z() = it.second[2];
                lm.flag = LandmarkFlag::ESTIMATED;
            }
            estimated_landmark_size ++;
        }
    }
}

std::vector<LandmarkPerId> D2LandmarkManager::getInitializedLandmarks() const {
    const Guard lock(state_lock);
    std::vector<LandmarkPerId> lm_per_frame_vec;
    for (auto it : landmark_db) {
        auto & lm = it.second;
        if (lm.track.size() >= params->landmark_estimate_tracks && lm.flag >= LandmarkFlag::INITIALIZED) {
            lm_per_frame_vec.push_back(lm);
        }
    }
    return lm_per_frame_vec;
}

LandmarkPerId & D2LandmarkManager::getLandmark(LandmarkIdType landmark_id) {
    const Guard lock(state_lock);
    return landmark_db.at(landmark_id);
}

std::vector<LandmarkPerId> D2LandmarkManager::getRelatedLandmarks(FrameIdType frame_id) const {
    const Guard lock(state_lock);
    if (related_landmarks.find(frame_id) == related_landmarks.end()) {
        return std::vector<LandmarkPerId>();
    }
    std::vector<LandmarkPerId> lm_per_frame_set;
    auto _landmark_ids = related_landmarks.at(frame_id);
    for (auto _id : _landmark_ids) {
        auto lm = landmark_db.at(_id);
        lm_per_frame_set.emplace_back(lm);
    }
    return lm_per_frame_set;
}

bool D2LandmarkManager::hasLandmark(LandmarkIdType landmark_id) const {
    const Guard lock(state_lock);
    return landmark_db.find(landmark_id) != landmark_db.end();
}

void D2LandmarkManager::removeLandmark(const LandmarkIdType & id) {
    landmark_db.erase(id);
    landmark_state.erase(id);
}

double triangulatePoint3DPts(std::vector<Swarm::Pose> poses, std::vector<Vector3d> &points, Vector3d &point_3d) {
    MatrixXd design_matrix(poses.size()*2, 4);
    assert(poses.size() > 0 && poses.size() == points.size() && "We at least have 2 poses and number of pts and poses must equal");
    for (unsigned int i = 0; i < poses.size(); i ++) {
        double p0x = points[i][0];
        double p0y = points[i][1];
        double p0z = points[i][2];
        Eigen::Matrix<double, 3, 4> pose;
        auto R0 = poses[i].att().toRotationMatrix();
        auto t0 = poses[i].pos();
        pose.leftCols<3>() = R0.transpose();
        pose.rightCols<1>() = -R0.transpose() * t0;
        design_matrix.row(i*2) = p0x * pose.row(2) - p0z*pose.row(0);
        design_matrix.row(i*2+1) = p0y * pose.row(2) - p0z*pose.row(1);
    }
    Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);

    MatrixXd pts(4, 1);
    pts << point_3d.x(), point_3d.y(), point_3d.z(), 1;
    MatrixXd errs = design_matrix*pts;
    return errs.norm()/ errs.rows(); 
}
}