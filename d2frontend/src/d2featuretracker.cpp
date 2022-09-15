#include <d2frontend/d2featuretracker.h>
#include <camodocal/camera_models/Camera.h>
#include <d2frontend/CNN/superglue_onnx.h>

#define MIN_HOMOGRAPHY 6
#define PYR_LEVEL 3
#define WIN_SIZE cv::Size(21, 21)

namespace D2FrontEnd {
D2FeatureTracker::D2FeatureTracker(D2FTConfig config):
    _config(config) {
    lmanager = new LandmarkManager;
    if (config.enable_superglue_local || config.enable_superglue_remote) {
        superglue = new SuperGlueOnnx(config.superglue_model_path);
    }
}

bool D2FeatureTracker::trackLocalFrames(VisualImageDescArray & frames) {
    const Guard lock(track_lock);
    bool iskeyframe = false;
    frame_count ++;
    TrackReport report;

    TicToc tic;
    if (!inited) {
        inited = true;
        ROS_INFO("[D2FeatureTracker] receive first, will init kf\n");
        processKeyframe(frames);
    }

    if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
        report.compose(track(frames.images[0]));
        report.compose(track(frames.images[0], frames.images[1]));
    } else if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
        for (auto & frame : frames.images) {
            report.compose(track(frame));
        }
    } else if(params->camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
        report.compose(track(frames.images[0]));
        report.compose(track(frames.images[1]));
        report.compose(track(frames.images[2]));
        report.compose(track(frames.images[3]));
        report.compose(track(frames.images[0], frames.images[1], true, LEFT_RIGHT_IMG_MATCH));
        report.compose(track(frames.images[1], frames.images[2], true, LEFT_RIGHT_IMG_MATCH));
        report.compose(track(frames.images[2], frames.images[3], true, LEFT_RIGHT_IMG_MATCH));
        report.compose(track(frames.images[0], frames.images[3], true, RIGHT_LEFT_IMG_MATCH));
    }
    
    if (isKeyframe(report)) {
        iskeyframe = true;
        processKeyframe(frames);
    }

    report.ft_time = tic.toc();
    // printf("[D2FeatureTracker::trackLocalFrames] Landmark mem : %.1fkB\n", (lmanager->total_lm_per_frame_num*sizeof(LandmarkPerFrame))/1024.0/1024.0);
    if (params->show) {
        if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
            draw(frames.images[0], frames.images[1], iskeyframe, report);
        } else if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
            for (auto & frame : frames.images) {
                draw(frame, iskeyframe, report);
            }
        } else if (params->camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
            draw(frames.images, iskeyframe, report);
        }
    }
    return iskeyframe;
}

bool D2FeatureTracker::trackRemoteFrames(VisualImageDescArray & frames) {
    const Guard lock(track_lock);
    bool matched = false;
    frame_count ++;
    TrackReport report;
    TicToc tic;
    if (params->camera_configuration == CameraConfig::STEREO_PINHOLE || params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
        report.compose(trackRemote(frames.images[0]));
        if (report.remote_matched_num > 0) {
            report.compose(trackRemote(frames.images[1], true));
        }
    } else {
        for (auto & frame : frames.images) {
            report.compose(trackRemote(frame));
        }
    }
    if (params->show) {
        if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
            drawRemote(frames.images[0], report);
        } else {
            for (auto & frame : frames.images) {
                drawRemote(frame, report);
            }
        }
    }
    report.ft_time = tic.toc();
    if (report.remote_matched_num > 0) {
        return true;
    } else {
        return false;
    }
}

TrackReport D2FeatureTracker::trackRemote(VisualImageDesc & frame, bool skip_whole_frame_match) {
    TrackReport report;
    if (current_keyframes.size() == 0) {
        printf("[D2FeatureTracker::trackRemote] waiting for initialization.\n");
        return report;
    }
    auto & current_keyframe = current_keyframes.back();
    if (!skip_whole_frame_match) {
        if (frame.image_desc.size() < NETVLAD_DESC_SIZE) {
            printf("[D2FeatureTracker::trackRemote] Warn: no vaild frame.image_desc.size() frame_id %ld ", frame.frame_id);
            return report;
        }
        const Map<VectorXf> vlad_desc_remote(frame.image_desc.data(), NETVLAD_DESC_SIZE);
        const Map<VectorXf> vlad_desc(current_keyframe.images[frame.camera_index].image_desc.data(), NETVLAD_DESC_SIZE);
        double netvlad_similar = vlad_desc.dot(vlad_desc_remote);
        if (netvlad_similar < params->vlad_threshold) {
            printf("[D2FeatureTracker::trackRemote] Remote image does not match current image %.2f/%.2f\n", netvlad_similar, params->vlad_threshold);
            return report;
        } else {
            // if (params->verbose)
                printf("[D2FeatureTracker::trackRemote] Remote image match current image %.2f/%.2f\n", netvlad_similar, params->vlad_threshold);
        }
    }

    if (current_keyframe.images.size() > 0 && current_keyframe.frame_id != frame.frame_id) {
        //Then current keyframe has been assigned, feature tracker by LK.
        auto & previous = current_keyframe.images[frame.camera_index];
        std::vector<int> ids_b_to_a;
        bool success = matchLocalFeatures(previous, frame, ids_b_to_a, _config.enable_superglue_remote);
        if (!success) {
            return report;
        }
        for (size_t i = 0; i < ids_b_to_a.size(); i++) { 
            if (ids_b_to_a[i] >= 0) {
                assert(ids_b_to_a[i] < previous.landmarkNum() && "too large");
                auto local_index = ids_b_to_a[i];
                auto &remote_lm = frame.landmarks[i];
                auto &local_lm = previous.landmarks[local_index];
                if (remote_lm.landmark_id >= 0 && local_lm.landmark_id>=0) {
                    if (local_to_remote.find(local_lm.landmark_id) == local_to_remote.end()) {
                        local_to_remote[local_lm.landmark_id] = std::unordered_map<int, LandmarkIdType>();
                    }
                    if (local_to_remote[local_lm.landmark_id].find(frame.drone_id) != local_to_remote[local_lm.landmark_id].end() && 
                        local_to_remote[local_lm.landmark_id][frame.drone_id] != remote_lm.landmark_id) {
                        // printf("[D2FeatureTracker::trackRemote] Possible ambiguous local landmark %ld for drone %ld prev matched to %ld now %ld \n",
                        //     local_lm.landmark_id, frame.drone_id, remote_lm.landmark_id, remote_lm.landmark_id);
                    }
                    remote_to_local[remote_lm.landmark_id] = local_lm.landmark_id;
                    // printf("[D2FeatureTracker::trackRemote] remote landmark %ld (prev %ld) -> local landmark %ld camera %ld \n",
                    //     remote_lm.landmark_id, local_to_remote[local_lm.landmark_id][frame.drone_id], local_lm.landmark_id, frame.camera_id);
                    local_to_remote[local_lm.landmark_id][frame.drone_id] = remote_lm.landmark_id;
                    remote_lm.landmark_id = local_lm.landmark_id;
                    if (_config.double_counting_common_feature || local_lm.stamp_discover < remote_lm.stamp_discover) {
                        remote_lm.solver_id = params->self_id;
                    } else {
                        remote_lm.solver_id = frame.drone_id;
                    }
                    // printf("[D2FeatureTracker::trackRemote] landmark %ld will solve by %ld\n",
                    //         remote_lm.landmark_id, remote_lm.solver_id);
                    report.remote_matched_num ++;
                }
            }
        }
    }
    // printf("[D2Frontend::D2FeatureTracker] match %d<->%d report.remote_matched_num %d",
    //     frame.drone_id, current_keyframe.drone_id, report.remote_matched_num);
    return report;
}

void D2FeatureTracker::cvtRemoteLandmarkId(VisualImageDesc & frame) const {
    int count = 0;
    for (auto & lm : frame.landmarks) {
        if (lm.landmark_id > 0 && remote_to_local.find(lm.landmark_id) != remote_to_local.end()) {
            // printf("Lm remote %ld -> %ld camera %ld\n", lm.landmark_id, remote_to_local.at(lm.landmark_id), lm.camera_id);
            lm.landmark_id = remote_to_local.at(lm.landmark_id);
            count ++;
        }
    }
    // printf("[D2FeatureTracker::cvtRemoteLandmarkId] Remote eff stereo %d\n", count);
}


TrackReport D2FeatureTracker::track(VisualImageDesc & frame) {
    TrackReport report;
    if (current_keyframes.size() > 0 && current_keyframes.back().frame_id != frame.frame_id) {
        auto & current_keyframe = current_keyframes.back();
        //Then current keyframe has been assigned, feature tracker by LK.
        auto & previous = current_keyframe.images[frame.camera_index];
        std::vector<int> ids_b_to_a;
        matchLocalFeatures(previous, frame, ids_b_to_a, _config.enable_superglue_local);
        for (size_t i = 0; i < ids_b_to_a.size(); i++) { 
            if (ids_b_to_a[i] >= 0) {
                assert(ids_b_to_a[i] < previous.spLandmarkNum() && "too large");
                auto prev_index = ids_b_to_a[i];
                auto landmark_id = previous.landmarks[prev_index].landmark_id;
                auto &cur_lm = frame.landmarks[i];
                auto &prev_lm = previous.landmarks[prev_index];
                cur_lm.landmark_id = landmark_id;
                cur_lm.velocity = cur_lm.pt3d_norm - prev_lm.pt3d_norm;
                cur_lm.velocity /= (frame.stamp - current_keyframe.stamp);
                cur_lm.stamp_discover = prev_lm.stamp_discover;
                lmanager->updateLandmark(cur_lm);
                report.sum_parallex += (prev_lm.pt3d_norm - cur_lm.pt3d_norm).norm();
                report.parallex_num ++;
                // printf("[D2FeatureTracker] landmark %d, prev_pt %f %f, cur_pt %f %f velocity %.2f %.2f %.2f norm prev_pt %f %f, cur_pt %f %f parallex %f\%\n", 
                //     landmark_id, 
                //     prev_lm.pt2d.x, prev_lm.pt2d.y,
                //     cur_lm.pt2d.x, cur_lm.pt2d.y,
                //     cur_lm.velocity.x(), cur_lm.velocity.y(), cur_lm.velocity.z(),
                //     prev_lm.pt3d_norm.x(), prev_lm.pt3d_norm.y(), cur_lm.pt3d_norm.x(), cur_lm.pt3d_norm.y(), 
                //     (prev_lm.pt3d_norm - cur_lm.pt3d_norm).norm()*100);
                if (lmanager->at(landmark_id).track.size() >= _config.long_track_frames) {
                    report.long_track_num ++;
                } else {
                    report.unmatched_num ++;
                }
            }
        }
    }
    if (_config.enable_lk_optical_flow) {
        //Enable LK optical flow feature tracker also.
        //This is for the case that the superpoint features is not tracked well.
        report.compose(trackLK(frame));
    }
    return report;
}

TrackReport D2FeatureTracker::trackLK(VisualImageDesc & frame) {
    //Track LK points
    TrackReport report;
    if (prev_lk_info.find(frame.camera_index) == prev_lk_info.end()) {
        prev_lk_info[frame.camera_index] = LKImageInfo();
    }
    auto cur_lk_pts = prev_lk_info[frame.camera_index].lk_pts;
    auto cur_lk_ids = prev_lk_info[frame.camera_index].lk_ids;
    if (!cur_lk_ids.empty())
        cur_lk_pts = opticalflowTrack(frame.raw_image, prev_lk_info[frame.camera_index].image, cur_lk_pts, cur_lk_ids);
    auto cur_all_pts = frame.landmarks2D();
    cur_all_pts.insert(cur_all_pts.end(), cur_lk_pts.begin(), cur_lk_pts.end());
    for (int i = 0; i < cur_lk_pts.size(); i++) {
        auto ret = createLKLandmark(frame, cur_lk_pts[i], cur_lk_ids[i]);
        if (!ret.first) {
            continue;
        }
        auto &lm = ret.second;
        auto track = lmanager->at(cur_lk_ids[i]).track;
        lm.velocity = extractPointVelocity(lm);
        auto prev_found = getPreviousLandmarkFrame(lm);
        if (prev_found.first) {
            lm.stamp_discover = prev_found.second.stamp_discover;
        }
        lmanager->updateLandmark(lm);
        frame.landmarks.emplace_back(lm);
        if (lmanager->at(cur_lk_ids[i]).track.size() >= _config.long_track_frames) {
            report.long_track_num ++;
        }
        //When computing parallex, we go to last keyframe
        if (!prev_found.first) {
            continue;
        }
        report.sum_parallex += (lm.pt3d_norm - prev_found.second.pt3d_norm).norm();
        report.parallex_num ++;
    }
    //Discover new points.
    std::vector<cv::Point2f> n_pts;
    if (!frame.raw_image.empty()) {
        detectPoints(frame.raw_image, n_pts, cur_all_pts, params->total_feature_num);
    } else {
        printf("[D2FeatureTracker::trackLK] empty image\n");
    }
    report.unmatched_num += n_pts.size();
    for (auto & pt : n_pts) {
        auto ret = createLKLandmark(frame, pt);
        if (!ret.first) {
            continue;
        }
        auto &lm = ret.second;
        auto _id = lmanager->addLandmark(lm);
        lm.landmark_id = _id;
        frame.landmarks.emplace_back(lm);
        cur_lk_pts.emplace_back(pt);
        cur_lk_ids.emplace_back(_id);
    }
    prev_lk_info[frame.camera_index].lk_pts = cur_lk_pts;
    prev_lk_info[frame.camera_index].lk_ids = cur_lk_ids;
    prev_lk_info[frame.camera_index].image  = frame.raw_image.clone();
    prev_lk_info[frame.camera_index].frame_id = frame.frame_id;
    return report;
}

std::pair<bool, LandmarkPerFrame > D2FeatureTracker::getPreviousLandmarkFrame(const LandmarkPerFrame & lpf) const {
    auto landmark_id = lpf.landmark_id;
    // printf("[D2FeatureTracker::extractPointVelocity] landmark_id %d\n", landmark_id);
    if (lmanager->hasLandmark(landmark_id) && lmanager->at(landmark_id).track.size() > 0) {
        auto lm_per_id = lmanager->at(landmark_id);
        for (int i = lm_per_id.track.size() - 1 ; i >= 0; i--) {
            auto lm = lm_per_id.track[i];
            if (lm.landmark_id == landmark_id && lm.frame_id != lpf.frame_id && lm.camera_id == lpf.camera_id) {
                return std::make_pair(true, lm);
            }
        }
    }
    LandmarkPerFrame lm;
    return std::make_pair(false, lm);
}

Vector3d D2FeatureTracker::extractPointVelocity(const LandmarkPerFrame & lpf) const {
    auto landmark_id = lpf.landmark_id;
    // printf("[D2FeatureTracker::extractPointVelocity] landmark_id %d\n", landmark_id);
    auto ret = getPreviousLandmarkFrame(lpf);
    if (ret.first) {
        auto lm = ret.second;
        Vector3d movement = lpf.pt3d_norm - lm.pt3d_norm;
        auto vel = movement / (lpf.stamp - lm.stamp);
        // printf("[D2FeatureTracker::extractPointVelocity] landmark %d, frame %d->%d, movement %f %f %f vel  %f %f %f \n", 
        //     landmark_id, lm.frame_id, lpf.frame_id, movement.x(), movement.y(), movement.z(), vel.x(), vel.y(), vel.z());
        return vel;
    }
    return Vector3d(0, 0, 0);
}

TrackReport D2FeatureTracker::track(const VisualImageDesc & left_frame, VisualImageDesc & right_frame, bool enable_lk, TrackLRType type) {
    auto prev_pts = left_frame.landmarks2D();
    auto cur_pts = right_frame.landmarks2D();
    std::vector<int> ids_b_to_a;
    TrackReport report;
    matchLocalFeatures(left_frame, right_frame, ids_b_to_a, _config.enable_superglue_local, type);
    for (size_t i = 0; i < ids_b_to_a.size(); i++) { 
        if (ids_b_to_a[i] >= 0) {
            assert(ids_b_to_a[i] < left_frame.spLandmarkNum() && "too large");
            auto prev_index = ids_b_to_a[i];
            auto landmark_id = left_frame.landmarks[prev_index].landmark_id;
            auto &cur_lm = right_frame.landmarks[i];
            auto &prev_lm = left_frame.landmarks[prev_index];
            cur_lm.landmark_id = landmark_id;
            cur_lm.stamp_discover = prev_lm.stamp_discover;
            cur_lm.velocity = extractPointVelocity(cur_lm);
            lmanager->updateLandmark(cur_lm);
            report.stereo_point_num ++;
        }
    }
    if (_config.enable_lk_optical_flow && enable_lk) {
        trackLK(left_frame, right_frame, type);
    }
    return report;
}

TrackReport D2FeatureTracker::trackLK(const VisualImageDesc & left_frame, VisualImageDesc & right_frame, TrackLRType type) {
    //Track LK points
    //This function MUST run after track(...)
    TrackReport report;
    auto cur_lk_pts = prev_lk_info[left_frame.camera_index].lk_pts;
    auto cur_lk_ids = prev_lk_info[left_frame.camera_index].lk_ids;
    assert(left_frame.frame_id == prev_lk_info[left_frame.camera_index].frame_id);
    if (!cur_lk_ids.empty())
        cur_lk_pts = opticalflowTrack(right_frame.raw_image, left_frame.raw_image, cur_lk_pts, cur_lk_ids, type);
    // printf("[trackLK] indices %d<->%d track type %d LK points: %lu\n", left_frame.camera_index, right_frame.camera_index, type, cur_lk_pts.size());
    for (int i = 0; i < cur_lk_pts.size(); i++) {
        auto ret = createLKLandmark(right_frame, cur_lk_pts[i], cur_lk_ids[i]);
        if (!ret.first) {
            continue;
        }
        auto &lm = ret.second;
        lm.stamp_discover = lmanager->at(cur_lk_ids[i]).stamp_discover;
        lm.velocity = extractPointVelocity(lm);
        lmanager->updateLandmark(lm);
        right_frame.landmarks.emplace_back(lm);
    }
    report.stereo_point_num = cur_lk_pts.size();
    return report;
}

bool D2FeatureTracker::isKeyframe(const TrackReport & report) {
    int prev_num = current_keyframes.size() > 0 ? current_keyframes.back().landmarkNum(): 0;
    if (report.meanParallex() > 0.5) {
        printf("[D2FeatureTracker] unexcepted mean parallex %f\n", report.meanParallex());
    }
    if (keyframe_count < _config.min_keyframe_num || 
        report.long_track_num < _config.long_track_thres ||
        prev_num < _config.last_track_thres ||
        report.unmatched_num > _config.new_feature_thres*prev_num || //Unmatched is assumed to be new
        report.meanParallex() > _config.parallex_thres) { //Attenion, if mismatch this will be big
        if (params->verbose) {
            printf("[D2FeatureTracker] keyframe_count: %d, long_track_num: %d, prev_num:%d, unmatched_num: %d, parallex: %f\n", 
                keyframe_count, report.long_track_num, prev_num, report.unmatched_num, report.meanParallex());
        }
            return true;
    }
    return false;
}

std::pair<bool, LandmarkPerFrame> D2FeatureTracker::createLKLandmark(const VisualImageDesc & frame, cv::Point2f pt, LandmarkIdType landmark_id) {
    Vector3d pt3d_norm;
    cams.at(frame.camera_index)->liftProjective(Eigen::Vector2d(pt.x, pt.y), pt3d_norm);
    pt3d_norm.normalize();
    if (pt3d_norm.hasNaN()) {
        return std::make_pair(false, LandmarkPerFrame());
    }
    LandmarkPerFrame lm = LandmarkPerFrame::createLandmarkPerFrame(landmark_id, frame.frame_id, frame.stamp, 
        LandmarkType::FlowLandmark, params->self_id, frame.camera_index, frame.camera_id, pt, pt3d_norm);
    if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
        //Add depth information
        auto dep = frame.raw_depth_image.at<unsigned short>(pt)/1000.0;
        if (dep > params->loopcamconfig->DEPTH_NEAR_THRES && dep < params->loopcamconfig->DEPTH_FAR_THRES) {
            auto pt3dcam = pt3d_norm*dep;
            lm.depth = pt3dcam.norm();
            lm.depth_mea = true;
        }
    }
    lm.color = extractColor(frame.raw_image, pt);
    return std::make_pair(true, lm);
}

void D2FeatureTracker::processKeyframe(VisualImageDescArray & frames) {
    if (current_keyframes.size() > 0 && current_keyframes.back().frame_id == frames.frame_id) {
        return;
    }
    keyframe_count ++;
    for (auto & frame: frames.images) {
        for (unsigned int i = 0; i < frame.landmarkNum(); i++) {
            if (frame.landmarks[i].landmark_id < 0) {
                if (params->camera_configuration == CameraConfig::STEREO_PINHOLE && frame.camera_index == 1) {
                    //We do not create new landmark for right camera
                    continue;
                }
                auto _id = lmanager->addLandmark(frame.landmarks[i]);
                frame.landmarks[i].setLandmarkId(_id);
            } else {
                lmanager->updateLandmark(frame.landmarks[i]);
            }
        }
    }
    // if (current_keyframe.frame_id >= 0) {
    //     lmanager->popFrame(current_keyframe.frame_id);
    // }
    current_keyframes.emplace_back(frames);
}

cv::Mat D2FeatureTracker::drawToImage(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report, bool is_right, bool is_remote) const {
    // ROS_INFO("Drawing ... %d", keyframe_count);
    cv::Mat img = frame.raw_image;
    int width = img.cols;
    auto & current_keyframe = current_keyframes.back();
    if (is_remote) {
        img = cv::imdecode(frame.image, cv::IMREAD_UNCHANGED);
        width = img.cols;
        if (img.empty()) {
            return cv::Mat();
        }
        cv::hconcat(img, current_keyframe.images[0].raw_image, img);
    }
    auto cur_pts = frame.landmarks2D();
    if (img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    char buf[64] = {0};
    int stereo_num = 0;
    for (size_t j = 0; j < cur_pts.size(); j++) {
        cv::Scalar color = cv::Scalar(0, 140, 255);
        if (frame.landmarks[j].type == SuperPointLandmark) {
            color = cv::Scalar(255, 0, 0); //Superpoint blue
        }
        cv::circle(img, cur_pts[j], 2, color, 2);
        auto _id = frame.landmarks[j].landmark_id;
        if (!lmanager->hasLandmark(_id)) {
            continue;
        }
        auto lm = lmanager->at(_id);
        if (_id >= 0) {
            cv::Point2f prev;
            bool prev_found = false;
            if (!lmanager->hasLandmark(_id)) {
                continue;
            }
            auto & pts2d = lmanager->at(_id).track;
            if (pts2d.size() == 0) 
                continue;
            if (is_right) {
                prev = pts2d.back().pt2d;
                prev_found = true;
            } else {
                for (int  index = pts2d.size()-1; index >= 0; index--) {
                    if (pts2d[index].camera_id == frame.camera_id && pts2d[index].frame_id != frame.frame_id) {
                        prev = lmanager->at(_id).track[index].pt2d;
                        prev_found = true;
                        break;
                    }
                }
            }
            if (!prev_found) {
                continue;
            }
            if (is_remote) {
                cv::line(img, prev + cv::Point2f(width, 0), cur_pts[j], cv::Scalar(0, 255, 0));
            } else {
                cv::arrowedLine(img, prev, cur_pts[j], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
            }
        }
        if (frame.landmarks[j].landmark_id >= 0) {
            stereo_num++;
        }
        if (_config.show_feature_id && frame.landmarks[j].landmark_id >= 0) {
            sprintf(buf, "%d", frame.landmarks[j].landmark_id%MAX_FEATURE_NUM);
            cv::putText(img, buf, cur_pts[j] - cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 1, color, 1);
        }
    }
    cv::Scalar color = cv::Scalar(255, 0, 0);
    // if (is_keyframe) {
    //     color = cv::Scalar(0, 0, 255);
    // }
    // if (is_right) {
    //     sprintf(buf, "Stereo points: %d", stereo_num);
    //     cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    // } else if (is_remote) {
    //     sprintf(buf, "Drone %d<->%d Matched points: %d", params->self_id, frame.drone_id, report.remote_matched_num);
    //     cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    // }
    // else {
    //     sprintf(buf, "KF/FRAME %d/%d @CAM %d ISKF: %d", keyframe_count, frame_count, 
    //         frame.camera_index, is_keyframe);
    //     cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    //     sprintf(buf, "TRACK %.1fms NUM %d LONG %d Parallex %.1f\%/%.1f",
    //         report.ft_time, report.parallex_num, report.long_track_num, report.meanParallex()*100, _config.parallex_thres*100);
    //     cv::putText(img, buf, cv::Point2f(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    // }
    return img;
}

void D2FeatureTracker::drawRemote(VisualImageDesc & frame, const TrackReport & report) const { 
    cv::Mat img = drawToImage(frame, false, report, false, true);
    if (img.empty()) {
        printf("[D2FeatureTracker::drawRemote] Unable to draw remote image, empty image found\n");
        return;
    }
    char buf[64] = {0};
    sprintf(buf, "RemoteMatched @ Drone %d", params->self_id);
    cv::imshow(buf, img);
    cv::waitKey(1);
    if (_config.write_to_file) {
        sprintf(buf, "%s/featureTracker_remote%06d.jpg", _config.output_folder.c_str(), frame_count);
        cv::imwrite(buf, img);
    }
}

void D2FeatureTracker::draw(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report) const {
    cv::Mat img = drawToImage(frame, is_keyframe, report);
    char buf[64] = {0};
    sprintf(buf, "featureTracker @ Drone %d", params->self_id);
    cv::imshow(buf, img);
    cv::waitKey(1);
    if (_config.write_to_file) {
        sprintf(buf, "%s/featureTracker%06d.jpg", _config.output_folder.c_str(), frame_count);
        cv::imwrite(buf, img);
    }
}

void D2FeatureTracker::draw(VisualImageDesc & lframe, VisualImageDesc & rframe, bool is_keyframe, const TrackReport & report) const {
    cv::Mat img = drawToImage(lframe, is_keyframe, report);
    cv::Mat img_r = drawToImage(rframe, is_keyframe, report, true);
    cv::hconcat(img, img_r, img);
    char buf[64] = {0};
    sprintf(buf, "featureTracker @ Drone %d", params->self_id);
    cv::imshow(buf, img);
    cv::waitKey(1);
    if (_config.write_to_file) {
        sprintf(buf, "%s/featureTracker%06d.jpg", _config.output_folder.c_str(), frame_count);
        cv::imwrite(buf, img);
    }
}

void D2FeatureTracker::draw(std::vector<VisualImageDesc> frames, bool is_keyframe, const TrackReport & report) const {
    cv::Mat img = drawToImage(frames[0], is_keyframe, report);
    cv::Mat img_r = drawToImage(frames[2], is_keyframe, report);
    cv::hconcat(img, img_r, img);
    cv::Mat img1 = drawToImage(frames[1], is_keyframe, report);
    cv::Mat img1_r = drawToImage(frames[3], is_keyframe, report);
    cv::hconcat(img1, img1_r, img1);
    cv::hconcat(img, img1, img);
    
    char buf[64] = {0};
    sprintf(buf, "featureTracker @ Drone %d", params->self_id);
    cv::imshow(buf, img);
    cv::waitKey(1);
    if (_config.write_to_file) {
        sprintf(buf, "%s/featureTracker%06d.jpg", _config.output_folder.c_str(), frame_count);
        cv::imwrite(buf, img);
    }
}

std::vector<float> getFeatureHalfImg(const std::vector<cv::Point2f> & pts, const std::vector<float> & desc, bool require_left, 
        std::map<int, int> & tmp_to_idx) {
    std::vector<float> desc_half;
    desc_half.reserve(desc.size());
    int c = 0;
    float move_cols = params->width_undistort *90.0/params->undistort_fov; //slightly lower than 0.5 cols when fov=200
    for (int i = 0; i < pts.size(); i++) {
        if (require_left && pts[i].x < params->width_undistort - move_cols || !require_left && pts[i].x >= move_cols) {
            tmp_to_idx[c] = i;
            //Copy from desc to desc_half
            std::copy(desc.begin() + i*FEATURE_DESC_SIZE, desc.begin() + (i+1)*FEATURE_DESC_SIZE, desc_half.begin() + c*FEATURE_DESC_SIZE);
            c += 1;
        }
    }
    return desc_half;
}

bool D2FeatureTracker::matchLocalFeatures(const VisualImageDesc & img_desc_a, const VisualImageDesc & img_desc_b, 
        std::vector<int> & ids_b_to_a, bool enable_superglue, TrackLRType type) {
    TicToc tic;
    auto & raw_desc_a = img_desc_a.landmark_descriptor;
    auto & raw_desc_b = img_desc_b.landmark_descriptor;
    auto pts_a = img_desc_a.landmarks2D(true);
    auto pts_b = img_desc_b.landmarks2D(true);
    auto pts_a_normed = img_desc_a.landmarks2D(true, true);
    auto pts_b_normed = img_desc_b.landmarks2D(true, true);

    std::vector<int> ids_a, ids_b;
    std::vector<cv::DMatch> _matches;
    ids_b_to_a.resize(pts_b.size());
    std::fill(ids_b_to_a.begin(), ids_b_to_a.end(), -1);
    if (enable_superglue) {
        //Superglue only support whole image matching
        auto & scores0 = img_desc_a.landmark_scores;
        auto & scores1 = img_desc_b.landmark_scores;
        _matches = superglue->inference(pts_a, pts_b, raw_desc_a, raw_desc_b, scores0, scores1);
    } else {
        if (type == WHOLE_IMG_MATCH) {
            const cv::Mat desc_a(raw_desc_a.size()/FEATURE_DESC_SIZE, FEATURE_DESC_SIZE, CV_32F, const_cast<float *>(raw_desc_a.data()));
            const cv::Mat desc_b(raw_desc_b.size()/FEATURE_DESC_SIZE, FEATURE_DESC_SIZE, CV_32F, const_cast<float *>(raw_desc_b.data()));
            cv::BFMatcher bfmatcher(cv::NORM_L2, true);
            bfmatcher.match(desc_a, desc_b, _matches); //Query train result
        } else {
            std::map<int, int> tmp_to_idx_a, tmp_to_idx_b;
            const auto desc_a_filtered = getFeatureHalfImg(pts_a, raw_desc_a, type==LEFT_RIGHT_IMG_MATCH, tmp_to_idx_a);
            const auto desc_b_filtered = getFeatureHalfImg(pts_b, raw_desc_b, type==RIGHT_LEFT_IMG_MATCH, tmp_to_idx_b);
            if (tmp_to_idx_a.size() == 0 || tmp_to_idx_b.size() == 0) {
                printf("[D2FeatureTracker] No feature to match.\n");
                return false;
            }
            cv::BFMatcher bfmatcher(cv::NORM_L2, true);
            const cv::Mat desc_a(tmp_to_idx_a.size(), FEATURE_DESC_SIZE, CV_32F, const_cast<float *>(desc_a_filtered.data()));
            const cv::Mat desc_b(tmp_to_idx_b.size(), FEATURE_DESC_SIZE, CV_32F, const_cast<float *>(desc_b_filtered.data()));
            bfmatcher.match(desc_a, desc_b, _matches); //Query train result
            for (auto & match : _matches) {
                match.queryIdx = tmp_to_idx_a[match.queryIdx];
                match.trainIdx = tmp_to_idx_b[match.trainIdx];
            }
        }
    }
    std::vector<cv::Point2f> matched_pts_a_normed, matched_pts_b_normed, matched_pts_a, matched_pts_b;
    for (auto match : _matches) {
        ids_a.push_back(match.queryIdx);
        ids_b.push_back(match.trainIdx);
        matched_pts_a_normed.push_back(pts_a_normed[match.queryIdx]);
        matched_pts_b_normed.push_back(pts_b_normed[match.trainIdx]);
        matched_pts_a.push_back(pts_a[match.queryIdx]);
        matched_pts_b.push_back(pts_b[match.trainIdx]);
    }
    if (params->ftconfig->check_homography && !enable_superglue) {
        std::vector<unsigned char> mask;
        if (matched_pts_a_normed.size() < MIN_HOMOGRAPHY) {
            return false;
        }
        cv::findHomography(matched_pts_a, matched_pts_b, cv::RANSAC, params->ftconfig->ransacReprojThreshold, mask);
        reduceVector(ids_a, mask);
        reduceVector(ids_b, mask);
        reduceVector(matched_pts_a, mask);
        reduceVector(matched_pts_b, mask);
    }
    for (auto i = 0; i < ids_a.size(); i++) {
        if (ids_a[i] >= pts_a.size()) {
            printf("ids_a[i] > pts_a.size() why is this case?\n");
            continue;
        }
        ids_b_to_a[ids_b[i]] = ids_a[i];
    }
    
    // //Plot matches
    // std::vector<cv::KeyPoint> kps_a, kps_b;
    // //Kps from points
    // for (int i = 0; i < pts_a.size(); i++) {
    //     kps_a.push_back(cv::KeyPoint(pts_a[i].x, pts_a[i].y, 1));
    // }
    // for (int i = 0; i < pts_b.size(); i++) {
    //     kps_b.push_back(cv::KeyPoint(pts_b[i].x, pts_b[i].y, 1));
    // }
    // cv::Mat show;
    // cv::drawMatches(img_desc_a.raw_image, kps_a, img_desc_b.raw_image, kps_b, _matches, show);
    // cv::imshow("raw_matches", show);
    // cv::hconcat(img_desc_a.raw_image, img_desc_b.raw_image, show);
    // cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
    // for (int i = 0; i < matched_pts_a.size(); i++) {
    //     cv::line(show, matched_pts_a[i], matched_pts_b[i] + cv::Point2f(show.cols/2, 0), cv::Scalar(0, 255, 0), 1);
    // }
    // cv::imshow("filtered_matches", show);

    if (params->verbose || params->enable_perf_output)
        printf("[D2FeatureTracker::track] match local features %d:%d %.3f ms\n", pts_a.size(), pts_b.size(), tic.toc());
    if (ids_b.size() >= params->ftconfig->remote_min_match_num) {
        return true;
    }
    return false;
}

void detectPoints(const cv::Mat & img, std::vector<cv::Point2f> & n_pts, std::vector<cv::Point2f> & cur_pts, int require_pts) {
    int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());
    cv::Mat mask;
    if (params->enable_perf_output) {
        ROS_INFO("Lost %d pts; Require %d will detect %d", lack_up_top_pts, require_pts, lack_up_top_pts > require_pts/4);
    }
    if (lack_up_top_pts > require_pts/4) {
        cv::Mat d_prevPts;
        cv::goodFeaturesToTrack(img, d_prevPts, lack_up_top_pts, 0.01, params->feature_min_dist, mask);
        std::vector<cv::Point2f> n_pts_tmp;
        // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
        if(!d_prevPts.empty()) {
            n_pts_tmp = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
        }
        else {
            n_pts_tmp.clear();
        }
        n_pts.clear();
        std::vector<cv::Point2f> all_pts = cur_pts;
        for (auto & pt : n_pts_tmp) {
            bool has_nearby = false;
            for (auto &pt_j: all_pts) {
                if (cv::norm(pt-pt_j) < params->feature_min_dist) {
                    has_nearby = true;
                    break;
                }
            }
            if (!has_nearby) {
                n_pts.push_back(pt);
                all_pts.push_back(pt);
            }
        }
    }
    else {
        n_pts.clear();
    }
}  

bool inBorder(const cv::Point2f &pt, cv::Size shape)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < shape.width - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < shape.height - BORDER_SIZE;
}

std::vector<cv::Point2f> opticalflowTrack(const cv::Mat & cur_img, const cv::Mat & prev_img, std::vector<cv::Point2f> & prev_pts, 
        std::vector<LandmarkIdType> & ids, D2FeatureTracker::TrackLRType type) {
    if (prev_pts.size() == 0) {
        return std::vector<cv::Point2f>();
    }
    TicToc tic;
    std::vector<uchar> status;
    std::vector<cv::Point2f> cur_pts;
    float move_cols = cur_img.cols*90.0/params->undistort_fov; //slightly lower than 0.5 cols when fov=200

    if (prev_pts.size() == 0) {
        return std::vector<cv::Point2f>();
    }

    if (type == D2FeatureTracker::WHOLE_IMG_MATCH) {
        cur_pts = prev_pts;
    } else  {
        status.resize(prev_pts.size());
        std::fill(status.begin(), status.end(), 0);
        if (type == D2FeatureTracker::LEFT_RIGHT_IMG_MATCH) {
            for (unsigned int i = 0; i < prev_pts.size(); i++) {
                auto pt = prev_pts[i];
                if (pt.x < cur_img.cols - move_cols) {
                    pt.x += move_cols;
                    status[i] = 1;
                    cur_pts.push_back(pt);
                }
            }
        } else {
            for (unsigned int i = 0; i < prev_pts.size(); i++) {
                auto pt = prev_pts[i];
                if (pt.x >= move_cols) {
                    pt.x -= move_cols;
                    status[i] = 1;
                    cur_pts.push_back(pt);
                }
            }
        }
        reduceVector(prev_pts, status);
        reduceVector(ids, status);
    }
    status.resize(0);
    if (cur_pts.size() == 0) {
        return std::vector<cv::Point2f>();
    }
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = cur_pts;
    for (unsigned int i = 0; i < prev_pts.size(); i++) {
        auto & pt = reverse_pts[i];
        if (type == D2FeatureTracker::LEFT_RIGHT_IMG_MATCH && status[i] == 1) {
            pt.x -= move_cols;
        }
        if (type == D2FeatureTracker::RIGHT_LEFT_IMG_MATCH && status[i] == 1) {
            pt.x += move_cols;
        }
    }
    cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    // if (type == D2FeatureTracker::LEFT_RIGHT_IMG_MATCH) {
    //     cv::Mat show;
    //     cv::vconcat(prev_img, cur_img, show);
    //     for (unsigned int i = 0; i < cur_pts.size(); i++) {
    //         //Draw arrows on the flow field if status[i]
    //         if (status[i] && reverse_status[i] && cv::norm(prev_pts[i] - reverse_pts[i]) <= 0.5) {
    //             cv::Point2f prev_pt = prev_pts[i];
    //             cv::Point2f cur_pt = cur_pts[i];
    //             cv::Point2f reverse_pt = reverse_pts[i];
    //             // cv::Point2f reverse_diff = reverse_pt - cur_pt;
    //             cv::arrowedLine(show, prev_pt, cur_pt, cv::Scalar(0, 255, 0), 2);
    //             cv::arrowedLine(show, cur_pt, reverse_pt, cv::Scalar(0, 0, 255), 2);
    //         }
    //     }
    //     cv::imshow("opticalflowTrack", show);
    // }
    for(size_t i = 0; i < status.size(); i++)
    {
        if(status[i] && reverse_status[i] && cv::norm(prev_pts[i] - reverse_pts[i]) <= 0.5)
        {
            status[i] = 1;
        }
        else
            status[i] = 0;
    }

    for (int i = 0; i < int(cur_pts.size()); i++){
        if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
            status[i] = 0;
        }
    }   
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    return cur_pts;
} 

}