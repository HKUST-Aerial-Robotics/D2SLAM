#include <d2frontend/d2featuretracker.h>
#include <camodocal/camera_models/Camera.h>

namespace D2FrontEnd {
#define MIN_HOMOGRAPHY 6
#define PYR_LEVEL 3
#define WIN_SIZE cv::Size(21, 21)

bool D2FeatureTracker::trackLocalFrames(VisualImageDescArray & frames) {
    const Guard lock(state_lock);
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
    } else {
        for (auto & frame : frames.images) {
            report.compose(track(frame));
        }
    }
    
    if (isKeyframe(report)) {
        iskeyframe = true;
        processKeyframe(frames);
    }

    report.ft_time = tic.toc();

    if (params->show) {
        if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
            draw(frames.images[0], frames.images[1], iskeyframe, report);
        } else {
            for (auto & frame : frames.images) {
                draw(frame, iskeyframe, report);
            }
        }
    }
    return iskeyframe;
}

bool D2FeatureTracker::trackRemoteFrames(VisualImageDescArray & frames) {
    const Guard lock(state_lock);
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
    if (!skip_whole_frame_match) {
        if (current_keyframe.images.size() == 0) {
            printf("[D2FeatureTracker::trackRemote] waiting for initialization.\n");
            return report;
        }
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
            if (params->verbose)
                printf("[D2FeatureTracker::trackRemote] Remote image match current image %.2f/%.2f\n", netvlad_similar, params->vlad_threshold);
        }
    }

    if (current_keyframe.images.size() > 0 && current_keyframe.frame_id != frame.frame_id) {
        //Then current keyframe has been assigned, feature tracker by LK.
        auto & previous = current_keyframe.images[frame.camera_index];
        auto prev_pts = previous.landmarks2D();
        auto cur_pts = frame.landmarks2D();
        std::vector<int> ids_down_to_up;
        bool success = matchLocalFeatures(prev_pts, cur_pts, previous.landmark_descriptor, frame.landmark_descriptor, ids_down_to_up);
        if (!success) {
            return report;
        }
        for (size_t i = 0; i < ids_down_to_up.size(); i++) { 
            if (ids_down_to_up[i] >= 0) {
                assert(ids_down_to_up[i] < previous.landmarkNum() && "too large");
                auto local_index = ids_down_to_up[i];
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
                    if (remote_lm.stamp_discover < local_lm.stamp_discover) {
                        remote_lm.solver_id = frame.drone_id;
                    } else {
                        remote_lm.solver_id = params->self_id;
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
    if (current_keyframe.images.size() > 0 && current_keyframe.frame_id != frame.frame_id) {
        //Then current keyframe has been assigned, feature tracker by LK.
        auto & previous = current_keyframe.images[frame.camera_index];
        auto prev_pts = previous.landmarks2D();
        auto cur_pts = frame.landmarks2D();
        std::vector<int> ids_down_to_up;
        matchLocalFeatures(prev_pts, cur_pts, previous.landmark_descriptor, frame.landmark_descriptor, ids_down_to_up);
        for (size_t i = 0; i < ids_down_to_up.size(); i++) { 
            if (ids_down_to_up[i] >= 0) {
                assert(ids_down_to_up[i] < previous.landmarkNum() && "too large");
                auto prev_index = ids_down_to_up[i];
                auto landmark_id = previous.landmarks[prev_index].landmark_id;
                auto &cur_lm = frame.landmarks[i];
                auto &prev_lm = previous.landmarks[prev_index];
                cur_lm.landmark_id = landmark_id;
                cur_lm.velocity = cur_lm.pt3d_norm - prev_lm.pt3d_norm;
                cur_lm.velocity /= (frame.stamp - current_keyframe.stamp);
                cur_lm.stamp_discover = prev_lm.stamp_discover;
                report.sum_parallex += (prev_lm.pt3d_norm - cur_lm.pt3d_norm).norm();
                report.parallex_num ++;
                // printf("[D2FeatureTracker] landmark %d, prev_pt %f %f, cur_pt %f %f norm prev_pt %f %f, cur_pt %f %f parallex %f\%\n", 
                //     landmark_id, 
                //     prev_lm.pt2d.x, prev_lm.pt2d.y,
                //     cur_lm.pt2d.x, cur_lm.pt2d.y,
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
    std::vector<cv::Point2f> n_pts;
    auto cur_all_pts = frame.landmarks2D();
    cur_all_pts.insert(cur_all_pts.end(), cur_lk_pts.begin(), cur_lk_pts.end());
    detectPoints(frame.raw_image, n_pts, cur_all_pts, params->total_feature_num);
    report.unmatched_num += n_pts.size();
    for (int i = 0; i < cur_lk_pts.size(); i++) {
        auto lm = createLKLandmark(frame, cur_lk_pts[i], cur_lk_ids[i]);
        auto track = lmanager->at(cur_lk_ids[i]).track;
        LandmarkPerFrame prev_lm = track.back();
        if (lm.frame_id != prev_lm.frame_id) {
            lm.velocity = lm.pt3d_norm - prev_lm.pt3d_norm;
            lm.velocity /= (lm.stamp - prev_lm.stamp);
        }
        lm.stamp_discover = prev_lm.stamp_discover;
        lmanager->updateLandmark(lm);
        frame.landmarks.emplace_back(lm);
        if (lmanager->at(cur_lk_ids[i]).track.size() >= _config.long_track_frames) {
            report.long_track_num ++;
        }
        //When computing parallex, we go to last keyframe
        bool find_in_keyframe = false;
        for (auto & lm_per_frame : track) {
            if (lm_per_frame.frame_id == current_keyframe.frame_id) {
                prev_lm = lm_per_frame;
                find_in_keyframe = true;
                break;
            }
        }
        if (!find_in_keyframe) {
            continue;
        }
        
        report.sum_parallex += (lm.pt3d_norm - prev_lm.pt3d_norm).norm();
        report.parallex_num ++;
    }
    for (auto & pt : n_pts) {
        auto lm = createLKLandmark(frame, pt);
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

TrackReport D2FeatureTracker::track(const VisualImageDesc & left_frame, VisualImageDesc & right_frame) {
    auto prev_pts = left_frame.landmarks2D();
    auto cur_pts = right_frame.landmarks2D();
    std::vector<int> ids_down_to_up;
    TrackReport report;
    matchLocalFeatures(prev_pts, cur_pts, left_frame.landmark_descriptor, right_frame.landmark_descriptor, ids_down_to_up);
    for (size_t i = 0; i < ids_down_to_up.size(); i++) { 
        if (ids_down_to_up[i] >= 0) {
            assert(ids_down_to_up[i] < left_frame.landmarkNum() && "too large");
            auto prev_index = ids_down_to_up[i];
            auto landmark_id = left_frame.landmarks[prev_index].landmark_id;
            auto &cur_lm = right_frame.landmarks[i];
            auto &prev_lm = left_frame.landmarks[prev_index];
            cur_lm.landmark_id = landmark_id;
            cur_lm.stamp_discover = prev_lm.stamp_discover;
            cur_lm.velocity = cur_lm.velocity;
            //TODO:Fix this for TD
            // if (lmanager->hasLandmark(landmark_id) && lmanager->at(landmark_id).track_r.size() > 0) {
            //     auto last_right = lmanager->at(landmark_id).track_r.back();
            //     if (left_frame.stamp - last_right.stamp < _config.max_pts_velocity_time) {
            //         cur_lm.velocity = last_right.pt3d_norm - cur_lm.pt3d_norm;
            //         cur_lm.velocity /= (last_right.stamp - left_frame.stamp);
            //     }
            // }
            report.stereo_point_num ++;
        }
    }
    if (_config.enable_lk_optical_flow) {
        trackLK(left_frame, right_frame);
    }
    return report;
}

TrackReport D2FeatureTracker::trackLK(const VisualImageDesc & left_frame, VisualImageDesc & right_frame) {
    //Track LK points
    //This function MUST run after track(...)
    TrackReport report;
    auto cur_lk_pts = prev_lk_info[left_frame.camera_index].lk_pts;
    auto cur_lk_ids = prev_lk_info[left_frame.camera_index].lk_ids;
    assert(left_frame.frame_id == prev_lk_info[left_frame.camera_index].frame_id);
    if (!cur_lk_ids.empty())
        cur_lk_pts = opticalflowTrack(right_frame.raw_image, left_frame.raw_image, cur_lk_pts, cur_lk_ids);
    for (int i = 0; i < cur_lk_pts.size(); i++) {
        auto lm = createLKLandmark(right_frame, cur_lk_pts[i], cur_lk_ids[i]);
        lm.stamp_discover = lmanager->at(cur_lk_ids[i]).stamp_discover;
        lmanager->updateLandmark(lm);
        right_frame.landmarks.emplace_back(lm);
        auto lm_per_id = lmanager->at(cur_lk_ids[i]);
        // TODO: Fix this for TD estimation
        // if (lm_per_id.track_r.size() > 0) {
        //     auto prev_lm = lm_per_id.track_r.back();
        //     if (lm.frame_id != prev_lm.frame_id) {
        //         lm.velocity = lm.pt3d_norm - prev_lm.pt3d_norm;
        //         lm.velocity /= (lm.stamp - prev_lm.stamp);
        //     }
        // }
    }
    report.stereo_point_num = cur_lk_pts.size();
    return report;
}

bool D2FeatureTracker::isKeyframe(const TrackReport & report) {
    int prev_num = current_keyframe.landmarkNum();
    if (report.meanParallex() > 0.5) {
        printf("[D2FeatureTracker] unexcepted mean parallex %f\n", report.meanParallex());
        exit(0);
    }
    if (keyframe_count < _config.min_keyframe_num || 
        report.long_track_num < _config.long_track_thres ||
        prev_num < _config.last_track_thres ||
        report.unmatched_num > _config.new_feature_thres*prev_num || //Unmatched is assumed to be new
        report.meanParallex() > _config.parallex_thres) { //Attenion, if mismatch this will be big
            return true;
    }
    return false;
}

LandmarkPerFrame D2FeatureTracker::createLKLandmark(const VisualImageDesc & frame, cv::Point2f pt, LandmarkIdType landmark_id) {
    Vector3d pt3d_norm;
    cams.at(frame.camera_index)->liftProjective(Eigen::Vector2d(pt.x, pt.y), pt3d_norm);
    pt3d_norm.normalize();
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
    return lm;
}

int LandmarkManager::addLandmark(const LandmarkPerFrame & lm) {
    auto _id = count + MAX_FEATURE_NUM*params->self_id;
    count ++;
    landmark_db[_id] = lm;
    return _id;
}

void LandmarkManager::updateLandmark(const LandmarkPerFrame & lm) {
    if (landmark_db.find(lm.landmark_id) == landmark_db.end()) {
        landmark_db[lm.landmark_id] = lm;
    } else {
        landmark_db.at(lm.landmark_id).add(lm);
    }
    assert(lm.landmark_id >= 0 && "landmark id must > 0");
}

void D2FeatureTracker::processKeyframe(VisualImageDescArray & frames) {
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
    current_keyframe = frames;
}

cv::Mat D2FeatureTracker::drawToImage(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report, bool is_right, bool is_remote) const {
    // ROS_INFO("Drawing ... %d", keyframe_count);
    cv::Mat img = frame.raw_image;
    int width = img.cols;
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
        cv::Scalar color = cv::Scalar(0, 255, 255);
        cv::circle(img, cur_pts[j], 1, color, 2);
        auto _id = frame.landmarks[j].landmark_id;
        if (!lmanager->hasLandmark(_id)) {
            continue;
        }
        auto lm = lmanager->at(_id);
        if (_id >= 0) {
            cv::Point2f prev;
            if (!lmanager->hasLandmark(_id)) {
                continue;
            }
            auto & pts2d = lmanager->at(_id).track;
            if (pts2d.size() == 0) 
                continue;
            if (!is_keyframe || pts2d.size() < 2 || is_right) {
                prev = pts2d.back().pt2d;
            } else {
                for (int  index = pts2d.size()-2; index >= 0; index--) {
                    if (pts2d[index].camera_id == frame.camera_id) {
                        prev = lmanager->at(_id).track[index].pt2d;
                        break;
                    }
                }
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
            cv::putText(img, buf, cur_pts[j] - cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }
    cv::Scalar color = cv::Scalar(255, 0, 0);
    if (is_keyframe) {
        color = cv::Scalar(0, 0, 255);
    }
    if (is_right) {
        sprintf(buf, "Stereo points: %d", stereo_num);
        cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    } else if (is_remote) {
        sprintf(buf, "Drone %d<->%d Matched points: %d", params->self_id, frame.drone_id, report.remote_matched_num);
        cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
    else {
        sprintf(buf, "KF/FRAME %d/%d @CAM %d ISKF: %d", keyframe_count, frame_count, 
            frame.camera_index, is_keyframe);
        cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        sprintf(buf, "TRACK %.1fms NUM %d LONG %d Parallex %.1f\%/%.1f",
            report.ft_time, report.parallex_num, report.long_track_num, report.meanParallex()*100, _config.parallex_thres*100);
        cv::putText(img, buf, cv::Point2f(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

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

bool matchLocalFeatures(const std::vector<cv::Point2f> & pts_up, const std::vector<cv::Point2f> & pts_down, 
        const std::vector<float> & _desc_up, const std::vector<float> & _desc_down, 
        std::vector<int> & ids_down_to_up) {
    const cv::Mat desc_up(_desc_up.size()/FEATURE_DESC_SIZE, FEATURE_DESC_SIZE, CV_32F, const_cast<float *>(_desc_up.data()));
    const cv::Mat desc_down(_desc_down.size()/FEATURE_DESC_SIZE, FEATURE_DESC_SIZE, CV_32F, const_cast<float *>(_desc_down.data()));

    cv::BFMatcher bfmatcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> _matches;
    bfmatcher.match(desc_up, desc_down, _matches); //Query train result
    ids_down_to_up.resize(pts_down.size());
    std::fill(ids_down_to_up.begin(), ids_down_to_up.end(), -1);
    std::vector<cv::Point2f> up_2d, down_2d;
    std::vector<int> ids_up, ids_down;
    if (params->enable_perf_output) {
        printf("[D2Frontend] matchLocalFeatures %ld %ld: matches %ld ", pts_up.size(), pts_down.size(), _matches.size());
    }
    for (auto match : _matches) {
        if (match.distance < ACCEPT_SP_MATCH_DISTANCE) {
            ids_up.push_back(match.queryIdx);
            ids_down.push_back(match.trainIdx);
            up_2d.push_back(pts_up[match.queryIdx]);
            down_2d.push_back(pts_down[match.trainIdx]);
        } else {
            // std::cout << "Giveup match dis" << match.distance << std::endl;
        }
    }

    std::vector<unsigned char> mask;
    if (params->ftconfig->check_homography) {
        if (up_2d.size() < MIN_HOMOGRAPHY) {
            return false;
        }
        cv::findHomography(up_2d, down_2d, cv::RANSAC, params->ftconfig->ransacReprojThreshold, mask);
        reduceVector(ids_up, mask);
        reduceVector(ids_down, mask);
    }
    for (auto i = 0; i < ids_up.size(); i++) {
        ids_down_to_up[ids_down[i]] = ids_up[i];
    }
    if (ids_down.size() > params->ftconfig->remote_min_match_num) {
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

std::vector<cv::Point2f> opticalflowTrack(const cv::Mat & cur_img, const cv::Mat & prev_img, std::vector<cv::Point2f> & prev_pts, std::vector<LandmarkIdType> & ids) {
    if (prev_pts.size() == 0) {
        return std::vector<cv::Point2f>();
    }
    TicToc tic;
    std::vector<uchar> status;

    if (prev_pts.size() == 0) {
        return std::vector<cv::Point2f>();
    }

    // vector<cv::Point2f> cur_pts = get_predict_pts(ids, prev_pts, prediction_points);
    std::vector<cv::Point2f> cur_pts = prev_pts;

    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL);
    std::vector<cv::Point2f> reverse_pts;
    std::vector<uchar> reverse_status;
    cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL);

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