#include <d2frontend/d2featuretracker.h>

namespace D2FrontEnd {
#define MIN_HOMOGRAPHY 6
bool D2FeatureTracker::track(VisualImageDescArray & frames) {
    bool iskeyframe = false;
    frame_count ++;
    TrackReport report;

    TicToc tic;
    if (!inited) {
        inited = true;
        ROS_INFO("[D2FeatureTracker] receive first, will init kf\n");
        processKeyframe(frames);
        return true;
    } else {
        for (auto & frame : frames.images) {
            report.compose(track(frame));
        }
        if (isKeyframe(report)) {
            iskeyframe = true;
            processKeyframe(frames);
        }
    }

    report.ft_time = tic.toc();

    if (params->debug_image) {
        for (auto & frame : frames.images) {
            draw(frame, iskeyframe, report);
        }
    }

    return iskeyframe;
}

TrackReport D2FeatureTracker::track(VisualImageDesc & frame) {
    auto & previous = current_keyframe.images[frame.camera_id];
    auto prev_pts = previous.landmarks2D();
    auto cur_pts = frame.landmarks2D();
    std::vector<int> ids_down_to_up;
    TrackReport report;
    matchLocalFeatures(prev_pts, cur_pts, previous.landmark_descriptor, frame.landmark_descriptor, ids_down_to_up);
    for (size_t i = 0; i < ids_down_to_up.size(); i++) { 
        if (ids_down_to_up[i] >= 0) {
            assert(ids_down_to_up[i] < previous.landmarkNum() && "too large");
            auto prev_index = ids_down_to_up[i];
            auto landmark_id = previous.landmarks[prev_index].landmark_id;
            auto &cur_lm = frame.landmarks[i];
            auto &prev_lm = previous.landmarks[prev_index];
            cur_lm.landmark_id = landmark_id;
            cur_lm.velocity = Vector3d(cur_lm.pt2d_norm.x() - prev_lm.pt2d_norm.x(), cur_lm.pt2d_norm.y() - prev_lm.pt2d_norm.y(), 0.);
            cur_lm.velocity /= (frame.stamp - current_keyframe.stamp);
            // printf("frame_count %d landmark_id %d prev pt %.2f %.2f cur pt %.2f %.2f vel %.2f %.2f\n", 
            //         frame_count,
            //         landmark_id,
            //         prev_lm.pt2d_norm.x(), prev_lm.pt2d_norm.y(), cur_lm.pt2d_norm.x(), cur_lm.pt2d_norm.y(), 
            //         cur_lm.velocity.x(), cur_lm.velocity.y());
            report.sum_parallex += (prev_lm.pt2d_norm - cur_lm.pt2d_norm).norm();
            report.parallex_num ++;
            if (lmanager->at(landmark_id).track.size() >= _config.long_track_frames) {
                report.long_track_num ++;
            } else {
                report.unmatched_num ++;
            }
        }
    }
    // printf("sum_parallex %f num %d mean %.2f\n", report.sum_parallex, report.parallex_num, report.meanParallex());
    return report;
}


bool D2FeatureTracker::isKeyframe(const TrackReport & report) {
    int prev_num = current_keyframe.landmarkNum();
    if (keyframe_count < _config.min_keyframe_num || 
        report.long_track_num < _config.long_track_thres ||
        prev_num < _config.last_track_thres ||
        report.unmatched_num > _config.new_feature_thres*prev_num || //Unmatched is assumed to be new
        report.meanParallex() > _config.parallex_thres) { //Attenion, if mismatch this will be big
            return true;
    }
    return false;
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
                auto _id = lmanager->addLandmark(frame.landmarks[i]);
                // printf("addLM frame_count %d landmark_id %d cur pt %.2f %.2f\n", 
                //     frame_count, _id, frame.landmarks[i].pt2d_norm.x(), frame.landmarks[i].pt2d_norm.y());
                frame.landmarks[i].setLandmarkId(_id);
            } else {
                lmanager->updateLandmark(frame.landmarks[i]);
            }
        }
    }
    current_keyframe = frames;
}


void D2FeatureTracker::draw(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report) {
    // ROS_INFO("Drawing ... %d", keyframe_count);
    cv::Mat img = frame.raw_image;
    auto cur_pts = frame.landmarks2D();
    if (img.channels() == 1) {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    char buf[64] = {0};
    for (size_t j = 0; j < cur_pts.size(); j++) {
        //Not tri
        //Not solving
        //Just New point yellow
        cv::Scalar color = cv::Scalar(0, 255, 255);
        // if (pts_status.find(ids[j]) != pts_status.end()) {
        //     int status = pts_status[ids[j]];
        //     if (status < 0) {
        //         //Removed points
        //         color = cv::Scalar(0, 0, 0);
        //     }
        //     if (status == 1) {
        //         //Good pt; But not used for solving; Blue 
        //         color = cv::Scalar(255, 0, 0);
        //     }
        //     if (status == 2) {
        //         //Bad pt; Red
        //         color = cv::Scalar(0, 0, 255);
        //     }
        //     if (status == 3) {
        //         //Good pt for solving; Green
        //         color = cv::Scalar(0, 255, 0);
        //     }
        // }

        cv::circle(img, cur_pts[j], 1, color, 2);
        auto _id = frame.landmarks[j].landmark_id;
        if (_id >= 0) {
            cv::Point2f prev;
            auto & pts2d = lmanager->at(_id).track;
            if (!is_keyframe || pts2d.size() < 2) {
                prev = pts2d.back().pt2d;
            } else {
                int index = pts2d.size()-2;
                prev = lmanager->at(_id).track[index].pt2d;
            }
            cv::arrowedLine(img, prev, cur_pts[j], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
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
    sprintf(buf, "KF/FRAME %d/%d @CAM %d ISKF: %d", keyframe_count, frame_count, 
        frame.camera_id, is_keyframe);
    cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    sprintf(buf, "TRACK %.1fms NUM %d LONG %d Parallex %.1f\%/%.1f",
        report.ft_time, report.parallex_num, report.long_track_num, report.meanParallex()*100, _config.parallex_thres*100);
    cv::putText(img, buf, cv::Point2f(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

    sprintf(buf, "featureTracker @ Drone %d", params->self_id);
    cv::imshow(buf, img);
    cv::waitKey(1);
    sprintf(buf, "/home/xuhao/output/images/featureTracker%06d.jpg", frame_count);
    cv::imwrite(buf, img);
}

void matchLocalFeatures(const std::vector<cv::Point2f> & pts_up, const std::vector<cv::Point2f> & pts_down, 
        std::vector<float> & _desc_up, std::vector<float> & _desc_down, 
        std::vector<int> & ids_down_to_up) {
    // printf("matchLocalFeatures %ld %ld: ", pts_up.size(), pts_down.size());
    const cv::Mat desc_up( _desc_up.size()/FEATURE_DESC_SIZE, FEATURE_DESC_SIZE, CV_32F, _desc_up.data());
    const cv::Mat desc_down( _desc_down.size()/FEATURE_DESC_SIZE, FEATURE_DESC_SIZE, CV_32F, _desc_down.data());

    cv::BFMatcher bfmatcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> _matches;
    bfmatcher.match(desc_up, desc_down, _matches); //Query train result
    ids_down_to_up.resize(pts_down.size());
    std::fill(ids_down_to_up.begin(), ids_down_to_up.end(), -1);
    std::vector<cv::Point2f> up_2d, down_2d;
    std::vector<int> ids_up, ids_down;
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
    if (up_2d.size() < MIN_HOMOGRAPHY) {
        return;
    }
    cv::findHomography(up_2d, down_2d, cv::RANSAC, 3, mask);
    reduceVector(ids_up, mask);
    reduceVector(ids_down, mask);
    for (auto i = 0; i < ids_up.size(); i++) {
        ids_down_to_up[ids_down[i]] = ids_up[i];
    }
    // printf("%ld/%ld matches...\n", ids_up.size(), _matches.size());
}

}