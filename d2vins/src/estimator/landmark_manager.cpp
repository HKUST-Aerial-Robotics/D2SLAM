#include "landmark_manager.hpp"
#include "d2vinsstate.hpp"
#include "../d2vins_params.hpp"

namespace D2VINS {

double triangulatePoint3DPts(const std::vector<Swarm::Pose> poses, const std::vector<Vector3d> &points, Vector3d &point_3d);

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

std::vector<LandmarkPerId> D2LandmarkManager::availableMeasurements(int max_pts, int max_solve_measurements, const std::set<FrameIdType> & current_frames) const {
    std::map<FrameIdType, int> current_landmark_num;
    std::map<FrameIdType, int> result_landmark_num;
    std::map<FrameIdType, std::set<D2Common::LandmarkIdType>> current_assoicated_landmarks;
    bool exit = false;
    std::set<D2Common::LandmarkIdType> ret_ids_set;
    std::vector<LandmarkPerId> ret_set;
    for (auto frame_id : current_frames) {
        current_landmark_num[frame_id] = 0;
        result_landmark_num[frame_id] = 0;
    }
    int count_measurements = 0;
    if (max_solve_measurements <= 0) {
        max_solve_measurements = 1000000;
    }
    while (!exit) {
        //found the frame with minimum landmarks in current frames
        if (current_landmark_num.size() == 0) {
            exit = true;
        }
        auto it = min_element(current_landmark_num.begin(), current_landmark_num.end(),
            [](decltype(current_landmark_num)::value_type& l, decltype(current_landmark_num)::value_type& r) -> 
                bool { return l.second < r.second; });
        auto frame_id = it->first;
        //Add the a landmark in its related landmarks with highest score
        if (related_landmarks.find(frame_id) == related_landmarks.end()) {
            //Remove the frame from current_landmark_num
            current_landmark_num.erase(frame_id);
            continue;
        }
        auto frame_related_landmarks = related_landmarks.at(frame_id);
        //Find the landmark with highest score
        LandmarkIdType lm_best;
        double score_best = -10000;
        bool found = false;
        for (auto & itre : frame_related_landmarks) {
            LandmarkIdType lm_id = itre.first;
            if (landmark_db.find(lm_id) == landmark_db.end() || ret_ids_set.find(lm_id) != ret_ids_set.end()) {
                //The landmark is not in the database or has been added
                continue;
            }
            auto & lm = landmark_db.at(lm_id);
            if (lm.track.size() >= params->landmark_estimate_tracks && 
                lm.flag >= LandmarkFlag::INITIALIZED) {
                if (lm.scoreForSolve(params->self_id) > score_best) {
                    score_best = lm.scoreForSolve(params->self_id);
                    lm_best = lm_id;
                    found = true;
                }
            }
        }
        if (found) {
            auto & lm = landmark_db.at(lm_best);
            ret_set.emplace_back(lm);
            ret_ids_set.insert(lm_best);
            count_measurements += lm.track.size();
            //Add the frame to current_landmark_num
            for (auto track: lm.track) {
                auto frame_id = track.frame_id;
                current_assoicated_landmarks[frame_id].insert(lm_best);
                //We count the landmark numbers, but not the measurements
                current_landmark_num[frame_id] = current_assoicated_landmarks[frame_id].size();
                result_landmark_num[frame_id] = current_landmark_num[frame_id];
            }
            if (ret_set.size() >= max_pts || count_measurements >= max_solve_measurements) {
                exit = true;
            }
        } else {
            //Remove the frame from current_landmark_num
            current_landmark_num.erase(frame_id);
        }
    }
    if (params->verbose) {
        printf("[D2VINS::D2LandmarkManager] Found %ld(total %ld) landmarks measure %d/%d in %ld frames\n", ret_set.size(), landmark_db.size(), 
                count_measurements, max_solve_measurements, result_landmark_num.size());
    }
    return ret_set;
}

double * D2LandmarkManager::getLandmarkState(LandmarkIdType landmark_id) const {
    const Guard lock(state_lock);
    return landmark_state.at(landmark_id);
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
    } else if (lm.track.size() >= params->landmark_estimate_tracks || lm.isMultiCamera()) {
        //Initialize by motion.
        std::vector<Swarm::Pose> poses;
        std::vector<Vector3d> points;
        auto ext_base = state->getExtrinsic(lm_first.camera_id);
        Eigen::Vector3d _min = (firstFrame.odom.pose()*ext_base).pos();
        Eigen::Vector3d _max = (firstFrame.odom.pose()*ext_base).pos();
        for (auto & it: lm.track) {
            auto frame = *state->getFramebyId(it.frame_id);
            auto ext = state->getExtrinsic(it.camera_id);
            auto cam_pose = frame.odom.pose()*ext;
            poses.push_back(cam_pose);
            points.push_back(it.pt3d_norm);
            _min = _min.cwiseMin((frame.odom.pose()*ext).pos());
            _max = _max.cwiseMax((frame.odom.pose()*ext).pos());
        }
        if ((_max - _min).norm() > params->depth_estimate_baseline) {
            //Initialize by triangulation
            Vector3d point_3d(0., 0., 0.);
            double tri_err = triangulatePoint3DPts(poses, points, point_3d);
            // printf("Lm %ld tri err %.3f thres %.3f\n", lm_id, tri_err*params->focal_length, params->tri_max_err*params->focal_length);
            if (tri_err < params->tri_max_err) {
                lm.position = point_3d;
                if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                    auto ptcam = (firstFrame.odom.pose()*ext_base).inverse()*point_3d;
                    auto inv_dep = 1/ptcam.norm();
                    if (inv_dep > params->min_inv_dep) {
                        lm.flag = LandmarkFlag::INITIALIZED;
                        *landmark_state[lm_id] = inv_dep;
                        if (params->debug_print_states) {
                            printf("[D2VINS::D2LandmarkManager] Landmark %ld tracks %ld baseline %.2f by tri. P %.3f %.3f %.3f inv_dep %.3f err %.3f\n",
                                lm_id, lm.track.size(), (_max - _min).norm(), point_3d.x(), point_3d.y(), point_3d.z(), inv_dep, tri_err);
                        }
                    } else {
                        lm.flag = LandmarkFlag::INITIALIZED;
                        *landmark_state[lm_id] = params->min_inv_dep;
                        if (params->debug_print_states) {
                            printf("\033[0;31m [D2VINS::D2LandmarkManager] Initialize failed too far away: landmark %ld tracks %ld baseline %.2f by triangulation position %.3f %.3f %.3f inv_dep %.3f \033[0m\n",
                                lm_id, lm.track.size(), (_max - _min).norm(), point_3d.x(), point_3d.y(), point_3d.z(), inv_dep);
                        }
                    }
                    // for (auto & it: lm.track) {
                    //     auto frame = *state->getFramebyId(it.frame_id);
                    //     auto ext = state->getExtrinsic(it.camera_id);
                    //     auto cam_pose = frame.odom.pose()*ext;
                    //     auto reproject_pos = cam_pose.inverse()*point_3d;
                    //     reproject_pos.normalize();
                    //     printf("Frame %ld camera_id %d index %d cam pose: %s pt3d norm %.3f %.3f %.3f reproject %.3f %.3f %.3f\n", 
                    //             it.frame_id, it.camera_id, it.camera_index, cam_pose.toStr().c_str(), it.pt3d_norm.x(), it.pt3d_norm.y(), it.pt3d_norm.z(), 
                    //             reproject_pos.x(), reproject_pos.y(), reproject_pos.z());
                    // }
                } else {
                    lm.flag = LandmarkFlag::INITIALIZED;
                    memcpy(landmark_state[lm_id], lm.position.data(), sizeof(state_type)*POS_SIZE);
                }
                // Some debug code
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
        if (lm.flag < LandmarkFlag::ESTIMATED) {
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
        printf("[D2VINS::D2LandmarkManager] Total %d initialized %d\n", 
            landmark_db.size(), inited_count);
    }
}

void D2LandmarkManager::outlierRejection(const D2EstimatorState * state, const std::set<LandmarkIdType> & used_landmarks) {
    const Guard lock(state_lock);
    int remove_count = 0;
    int total_count = 0;
    if (estimated_landmark_size < params->perform_outlier_rejection_num) {
        return;
    }
    for (auto & it: landmark_db) {
        auto & lm = it.second;
        auto lm_id = it.first;
        if(lm.flag == LandmarkFlag::ESTIMATED && used_landmarks.find(lm_id)!=used_landmarks.end()) {
            double err_sum = 0;
            double err_cnt = 0;
            int count_err_track = 0;
            total_count ++;
            for (auto it = lm.track.begin() + 1; it != lm.track.end();) {
                auto pose = state->getFramebyId(it->frame_id)->odom.pose();
                auto ext = state->getExtrinsic(it->camera_id);
                auto pt3d_n = it->pt3d_norm;
                Vector3d pos_cam = (pose*ext).inverse()*lm.position;
                pos_cam.normalize();
                //Compute reprojection error
                Vector3d reproj_error = pt3d_n - pos_cam;
                if (reproj_error.norm() * params->focal_length > params->landmark_outlier_threshold) {
                    count_err_track += 1;
                    // printf("[outlierRejection] remove outlier track LM %d frame %ld inv_dep/dep %.2f/%.2f reproj_err %.2f/%.2f\n",
                    //         lm_id, it->frame_id, *landmark_state[lm_id], 1./(*landmark_state[lm_id]), reproj_error.norm() * params->focal_length, 
                    //         params->landmark_outlier_threshold);
                    // //Remove the track
                    // it = lm.track.erase(it);
                    ++it;
                } else {
                    ++it;
                }
                err_sum += reproj_error.norm();
                err_cnt += 1;
            }
            lm.num_outlier_tracks = count_err_track;
            if (err_cnt > 0) {
                double reproj_err = err_sum/err_cnt;
                if (reproj_err*params->focal_length > params->landmark_outlier_threshold) {
                    remove_count ++;
                    lm.flag = LandmarkFlag::OUTLIER;
                    if (params->verbose) {
                        printf("[outlierRejection] remove LM %d inv_dep/dep %.2f/%.2f pos %.2f %.2f %.2f reproj_error %.2f\n",
                            lm_id, *landmark_state[lm_id], 1./(*landmark_state[lm_id]), lm.position.x(), lm.position.y(), lm.position.z(), reproj_err*params->focal_length);
                    }
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
                if (inv_dep < 0) {
                    printf("[Warn] negative inv dep %.2f found\n", inv_dep);
                }
                if (inv_dep < params->min_inv_dep) {
                    inv_dep = params->min_inv_dep;
                }
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

void D2LandmarkManager::removeLandmark(const LandmarkIdType & id) {
    landmark_db.erase(id);
    landmark_state.erase(id);
}

double triangulatePoint3DPts(const std::vector<Swarm::Pose> poses, const std::vector<Vector3d> &points, Vector3d &point_3d) {
    MatrixXd design_matrix(poses.size()*2, 4);
    assert(poses.size() > 0 && poses.size() == points.size() && "We at least have 2 poses and number of pts and poses must equal");
    for (unsigned int i = 0; i < poses.size(); i ++) {
        double p0x = points[i][0];
        double p0y = points[i][1];
        double p0z = points[i][2];
        Eigen::Matrix<double, 3, 4> pose;
        auto R0 = poses[i].R();
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

    double sum_err = 0;
    double err_pose_0 = 0.0;
    for (unsigned int i = 0; i < poses.size(); i ++) {
        auto reproject_pos = poses[i].inverse()*point_3d;
        reproject_pos.normalize();
        Vector3d err = points[i] - reproject_pos;
        if (i == 0) {
            err_pose_0 = err.norm();
        }
        sum_err += err.norm();
    }
    return sum_err/ points.size() + err_pose_0; 
}
}
