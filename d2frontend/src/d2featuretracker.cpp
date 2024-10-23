#include <camodocal/camera_models/Camera.h>
#include <d2common/d2vinsframe.h>
#include <d2frontend/CNN/superglue_onnx.h>
#include <d2frontend/d2featuretracker.h>
#include <d2frontend/loop_cam.h>
#include <d2frontend/utils.h>
#include <spdlog/spdlog.h>

#include <opencv2/core/cuda.hpp>

#define MIN_HOMOGRAPHY 6
using D2Common::Utility::TicToc;

namespace D2FrontEnd {
D2FeatureTracker::D2FeatureTracker(D2FTConfig config) : _config(config) {
  lmanager = new LandmarkManager;
  if (config.enable_superglue_local || config.enable_superglue_remote) {
    superglue = new SuperGlueOnnx(config.superglue_model_path);
  }
  image_width = params->width;
  if (params->camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
    image_width = params->width_undistort;
  }
  if (!_config.enable_search_local_aera) {
    _config.search_local_max_dist = -1;
  }
  search_radius = _config.search_local_max_dist * image_width;
  reference_frame_id = params->self_id;
  _config.sp_track_use_lk =
      _config.sp_track_use_lk && params->loopcamconfig->superpoint_max_num > 0;
  SPDLOG_INFO("sp {} _config.sp_track_use_lk {}",
              params->loopcamconfig->superpoint_max_num,
              _config.sp_track_use_lk);
}

void D2FeatureTracker::updatebySldWin(const std::vector<VINSFrame *> sld_win) {
  // update by sliding window
  const Guard lock(keyframe_lock);
  const Guard guard2(lmanager_lock);
  if (current_keyframes.size() == 0 || sld_win.size() == 0) return;
  std::map<FrameIdType, Swarm::Pose> sld_win_poses;
  for (auto &frame : sld_win) {
    sld_win_poses[frame->frame_id] = frame->odom.pose();
  }
  reference_frame_id = sld_win.back()->reference_frame_id;
  // Remove the keyframe not in the sliding window except the last one
  for (auto it = current_keyframes.begin(); it != current_keyframes.end();) {
    if (sld_win_poses.find(it->frame_id) == sld_win_poses.end() &&
        it->frame_id != current_keyframes.back().frame_id) {
      if (current_keyframes.size() <= 1) {
        it++;
      } else {
        lmanager->popFrame(it->frame_id);
        it = current_keyframes.erase(it);
      }
    } else {
      if (sld_win_poses.find(it->frame_id) != sld_win_poses.end()) {
        it->pose_drone = sld_win_poses[it->frame_id];
        for (auto &img : it->images) {
          img.pose_drone = sld_win_poses[it->frame_id];
        }
      }
      it++;
    }
  }
}

void D2FeatureTracker::updatebyLandmarkDB(
    const std::map<LandmarkIdType, LandmarkPerId> &vins_landmark_db) {
  // update by sliding window
  const Guard guard2(lmanager_lock);
  if (_config.enable_motion_prediction_local ||
      _config.enable_search_local_aera_remote) {
    auto &db = lmanager->getLandmarkDB();
    for (auto &kv : vins_landmark_db) {
      if (db.find(kv.first) != db.end()) {
        auto &lm = lmanager->at(kv.first);
        lm.flag = kv.second.flag;
        lm.position = kv.second.position;
      }
    }
  }
}

bool D2FeatureTracker::trackLocalFrames(VisualImageDescArray &frames) {
  const Guard lock(track_lock);
  const Guard guard(keyframe_lock);
  const Guard guard2(lmanager_lock);
  bool iskeyframe = false;
  frame_count++;
  TrackReport report;
  landmark_predictions_viz.clear();
  frames.send_to_backend = (frame_count % _config.frame_step) == 0;
  TicToc tic;
  if (!inited) {
    inited = true;
    SPDLOG_INFO("receive first, will init kf");
    iskeyframe = true;
    if (!_config.sp_track_use_lk) {
      processFrame(frames, true);
    }
    frames.send_to_backend = true;
  }
  if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
    report.compose(track(frames.images[0], frames.motion_prediction));
    if (_config.lr_match_use_lk) {
      frames.images[1].landmarks.clear();
      frames.images[1].landmark_descriptor.clear();
      frames.images[1].landmark_scores.clear();
    }
    report.compose(track(frames.images[0], frames.images[1], true,
                         WHOLE_IMG_MATCH, _config.lr_match_use_lk));
  } else if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
    for (auto &frame : frames.images) {
      report.compose(track(frame));
    }
  } else if (params->camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
    report.compose(track(frames.images[0], frames.motion_prediction));
    report.compose(track(frames.images[1], frames.motion_prediction));
    report.compose(track(frames.images[2], frames.motion_prediction));
    report.compose(track(frames.images[3], frames.motion_prediction));
    report.compose(
        track(frames.images[0], frames.images[1], true, LEFT_RIGHT_IMG_MATCH));
    report.compose(
        track(frames.images[1], frames.images[2], true, LEFT_RIGHT_IMG_MATCH));
    report.compose(
        track(frames.images[2], frames.images[3], true, LEFT_RIGHT_IMG_MATCH));
    report.compose(
        track(frames.images[0], frames.images[3], true, RIGHT_LEFT_IMG_MATCH));
  }
  if (isKeyframe(report) && frames.send_to_backend) {
    iskeyframe = true;
  }
  processFrame(frames, iskeyframe);
  report.ft_time = tic.toc();
  SPDLOG_INFO(
      "frame_id: {} is_kf {}, landmark_num: {}/{}, mean_para {:.2f}%, "
      "time_cost: {:.1f}ms ",
      frames.frame_id, iskeyframe, report.parallex_num, frames.landmarkNum(),
      report.meanParallex() * 100, report.ft_time);
  if (params->show) {
    if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
      draw(frames.images[0], frames.images[1], iskeyframe, report);
    } else if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
      for (auto &frame : frames.images) {
        draw(frame, iskeyframe, report);
      }
    } else if (params->camera_configuration ==
               CameraConfig::FOURCORNER_FISHEYE) {
      draw(frames, iskeyframe, report);
    }
  }
  if (report.stereo_point_num > _config.min_stereo_points) {
    frames.is_stereo = true;
  }
  return iskeyframe;
}

bool D2FeatureTracker::getMatchedPrevKeyframe(
    const VisualImageDescArray &frame_a, VisualImageDescArray &prev, int &dir_a,
    int &dir_b) {
  const Guard lock(keyframe_lock);
  if (current_keyframes.size() == 0) {
    return false;
  }
  if (params->camera_configuration == CameraConfig::STEREO_PINHOLE ||
      params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
    // Reverse seach in current keyframes
    for (int i = current_keyframes.size() - 1; i >= 0; i--) {
      const auto &last = current_keyframes[i];
      if (frame_a.images.size() == 0 ||
          frame_a.images[0].image_desc.size() != params->netvlad_dims) {
        ROS_ERROR("No vaild frame.image_desc.size() frame_id {}",
                  frame_a.frame_id);
        return false;
      }
      const Map<const VectorXf> vlad_desc_remote(
          frame_a.images[0].image_desc.data(), params->netvlad_dims);
      const Map<const VectorXf> vlad_desc(last.images[0].image_desc.data(),
                                          params->netvlad_dims);
      double netvlad_similar = vlad_desc.dot(vlad_desc_remote);
      if (netvlad_similar < params->track_remote_netvlad_thres) {
        spdlog::debug(
            "D{} Remote image does not match current image {:.2f}/{:.2f}",
            params->self_id, netvlad_similar,
            params->track_remote_netvlad_thres);
      } else {
        spdlog::debug("D{} Remote image match image {}({}) {:.2f}/{:.2f}",
                      params->self_id, i, last.frame_id, netvlad_similar,
                      params->track_remote_netvlad_thres);
        prev = last;
        dir_a = 0;
        dir_b = 0;
        return true;
      }
    }
  }
  if (params->camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
    std::vector<int> dirs{2, 3, 0, 1};
    dir_a = 2;
    // printf("[D2FeatureTracker::getMatchedPrevKeyframe] Remote frame %ld view
    // 2/%ld: gdesc %ld\n", frame_a.frame_id, frame_a.images.size(),
    // frame_a.images[2].image_desc.size());
    const Map<const VectorXf> vlad_desc_remote(
        frame_a.images[2].image_desc.data(), params->netvlad_dims);
    for (int i = current_keyframes.size() - 1; i >= 0; i--) {
      const auto &last = current_keyframes[i];
      for (int j = 0; j < last.images.size(); j++) {
        const Map<const VectorXf> vlad_desc(
            last.images[dirs[j]].image_desc.data(), params->netvlad_dims);
        double netvlad_similar = vlad_desc.dot(vlad_desc_remote);
        if (netvlad_similar < params->track_remote_netvlad_thres) {
        } else {
          prev = last;
          dir_b = dirs[j];
          spdlog::debug(
              "D{} Remote image match image drone {}({}) dir {}:{} "
              "{:.2f}/{:.2f}",
              params->self_id, i, last.frame_id, dir_a, dir_b, netvlad_similar,
              params->track_remote_netvlad_thres);
          return true;
        }
      }
    }
  }
  return false;
}

bool D2FeatureTracker::trackRemoteFrames(VisualImageDescArray &frames) {
  const Guard lock(track_lock);
  if (frames.is_lazy_frame || frames.matched_frame >= 0) {
    return false;
  }
  bool matched = false;
  landmark_predictions_viz.clear();
  frame_count++;
  TrackReport report;
  TicToc tic;
  int dir_cur = 0, dir_prev = 0;
  VisualImageDescArray prev;
  bool succ = getMatchedPrevKeyframe(frames, prev, dir_cur, dir_prev);
  if (!succ) {
    return false;
  }
  bool use_motion_predict = frames.reference_frame_id == reference_frame_id &&
                            _config.enable_search_local_aera_remote;
  // printf("[D2FeatureTracker::trackRemoteFrames] frame %ld ref %d cur_ref %d
  // use_motion_predict %d\n",
  //         frames.frame_id, frames.reference_frame_id, reference_frame_id,
  //         use_motion_predict);
  if (params->camera_configuration == CameraConfig::STEREO_PINHOLE ||
      params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
    report.compose(trackRemote(frames.images[0], prev.images[0],
                               use_motion_predict, frames.pose_drone));
    if (report.remote_matched_num > 0 &&
        params->camera_configuration == CameraConfig::STEREO_PINHOLE &&
        !_config.lr_match_use_lk) {
      report.compose(trackRemote(frames.images[1], prev.images[1],
                                 use_motion_predict, frames.pose_drone));
    }
  } else if (params->camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
    int max_dirs = 4;
    std::vector<int> dirs_cur;
    std::vector<int> dirs_prev;
    for (int _dir_a = dir_cur; _dir_a < dir_cur + max_dirs; _dir_a++) {
      int dir_a = _dir_a % max_dirs;
      int dir_b =
          ((dir_prev - dir_cur + max_dirs) % max_dirs + _dir_a) % max_dirs;
      if (dir_a < frames.images.size() && dir_b < prev.images.size()) {
        if (prev.images[dir_b].spLandmarkNum() > 0 &&
            frames.images[dir_a].spLandmarkNum() > 0) {
          dirs_cur.push_back(dir_a);
          dirs_prev.push_back(dir_b);
        }
      }
    }
    for (size_t i = 0; i < dirs_cur.size(); i++) {
      int dir_cur = dirs_cur[i];
      int dir_prev = dirs_prev[i];
      // printf("[D2FeatureTracker::trackRemoteFrames] dir %d:%d\n", dir_cur,
      // dir_prev);
      report.compose(trackRemote(frames.images[dir_cur], prev.images[dir_prev],
                                 use_motion_predict, frames.pose_drone));
    }
  }
  if (params->show && params->send_whole_img_desc && params->send_img) {
    if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
      drawRemote(frames, report);
    }
  }
  report.ft_time = tic.toc();
  if (params->enable_perf_output)
    SPDLOG_INFO(
        "[D2FeatureTracker::trackRemoteFrames] frame {}, matched {}, time "
        "{:.2f}ms",
        frames.frame_id, report.remote_matched_num, report.ft_time);
  if (report.remote_matched_num > 0) {
    return true;
  } else {
    return false;
  }
}

TrackReport D2FeatureTracker::trackRemote(
    VisualImageDesc &frame, const VisualImageDesc &prev_frame,
    bool use_motion_predict, const Swarm::Pose &motion_prediction) {
  TrackReport report;
  if (current_keyframes.size() == 0) {
    SPDLOG_INFO(
        "[D2FeatureTracker::trackRemote] waiting for initialization.\n");
    return report;
  }
  if (prev_frame.frame_id != frame.frame_id) {
    // Then current keyframe has been assigned, feature tracker by LK.
    std::vector<int> ids_b_to_a;
    // TODO: use motion prediction
    MatchLocalFeatureParams match_param;
    match_param.enable_superglue = _config.enable_superglue_remote;
    match_param.enable_prediction = use_motion_predict;
    match_param.pose_b_prediction = motion_prediction;
    match_param.pose_a = prev_frame.pose_drone;
    match_param.type = WHOLE_IMG_MATCH;
    match_param.search_radius =
        search_radius * 2;  // search radius is 100% larger
    match_param.plot = false;
    bool success =
        matchLocalFeatures(prev_frame, frame, ids_b_to_a, match_param);
    if (!success) {
      SPDLOG_WARN("matchLocalFeatures failed");
      return report;
    }
    for (size_t i = 0; i < ids_b_to_a.size(); i++) {
      if (ids_b_to_a[i] >= 0) {
        assert(ids_b_to_a[i] < prev_frame.landmarkNum() && "too large");
        auto local_index = ids_b_to_a[i];
        auto &remote_lm = frame.landmarks[i];
        auto &local_lm = prev_frame.landmarks[local_index];
        if (remote_lm.landmark_id >= 0 && local_lm.landmark_id >= 0) {
          if (local_to_remote.find(local_lm.landmark_id) ==
              local_to_remote.end()) {
            local_to_remote[local_lm.landmark_id] =
                std::unordered_map<int, LandmarkIdType>();
          }
          if (local_to_remote[local_lm.landmark_id].find(frame.drone_id) !=
                  local_to_remote[local_lm.landmark_id].end() &&
              local_to_remote[local_lm.landmark_id][frame.drone_id] !=
                  remote_lm.landmark_id) {
            // printf("[D2FeatureTracker::trackRemote] Possible ambiguous local
            // landmark %ld for drone %ld prev matched to %ld now %ld \n",
            //     local_lm.landmark_id, frame.drone_id, remote_lm.landmark_id,
            //     remote_lm.landmark_id);
          }
          remote_to_local[remote_lm.landmark_id] = local_lm.landmark_id;
          // printf("[D2FeatureTracker::trackRemote] remote landmark %ld (prev
          // %ld) -> local landmark %ld camera %ld \n",
          //     remote_lm.landmark_id,
          //     local_to_remote[local_lm.landmark_id][frame.drone_id],
          //     local_lm.landmark_id, frame.camera_id);
          local_to_remote[local_lm.landmark_id][frame.drone_id] =
              remote_lm.landmark_id;
          remote_lm.landmark_id = local_lm.landmark_id;
          if (_config.double_counting_common_feature ||
              local_lm.stamp_discover < remote_lm.stamp_discover) {
            remote_lm.solver_id = params->self_id;
          } else {
            remote_lm.solver_id = frame.drone_id;
          }
          // printf("[D2FeatureTracker::trackRemote] landmark %ld will solve by
          // %ld stamp %.3f:%.3f\n",
          //         remote_lm.landmark_id, remote_lm.solver_id,
          //         local_lm.stamp_discover, remote_lm.stamp_discover);
          report.remote_matched_num++;
        }
      }
    }
  }
  spdlog::debug(
      "[D2Frontend::D2FeatureTracker] match {}@cam{}<->{}@cam{} "
      "report.remote_matched_num {}",
      frame.drone_id, frame.camera_index, prev_frame.drone_id,
      frame.camera_index, report.remote_matched_num);
  return report;
}

void D2FeatureTracker::cvtRemoteLandmarkId(VisualImageDesc &frame) const {
  int count = 0;
  for (auto &lm : frame.landmarks) {
    if (lm.landmark_id > 0 &&
        remote_to_local.find(lm.landmark_id) != remote_to_local.end()) {
      // printf("Lm remote %ld -> %ld camera %ld\n", lm.landmark_id,
      // remote_to_local.at(lm.landmark_id), lm.camera_id);
      lm.landmark_id = remote_to_local.at(lm.landmark_id);
      count++;
    }
  }
  // printf("[D2FeatureTracker::cvtRemoteLandmarkId] Remote eff stereo %d\n",
  // count);
}

TrackReport D2FeatureTracker::track(VisualImageDesc &frame,
                                    const Swarm::Pose &motion_prediction) {
  TrackReport report;
  if (!_config.sp_track_use_lk && current_keyframes.size() > 0 &&
      current_keyframes.back().frame_id != frame.frame_id) {
    const auto &base_frame = _config.track_from_keyframe
                                 ? getLatestKeyframe()
                                 : current_keyframes.back();
    const auto &base_kfframe = getLatestKeyframe();
    // Then current keyframe has been assigned, feature tracker by LK.
    auto &previous = base_frame.images[params->camera_seq[frame.camera_index]];
    std::vector<int> ids_b_to_a;
    MatchLocalFeatureParams match_param;
    match_param.enable_superglue = _config.enable_superglue_local;
    match_param.enable_prediction = _config.enable_motion_prediction_local;
    match_param.pose_a = previous.pose_drone;
    match_param.pose_b_prediction = motion_prediction;
    match_param.search_radius = search_radius;
    match_param.enable_search_in_local = true;
    matchLocalFeatures(previous, frame, ids_b_to_a, match_param);
    for (size_t i = 0; i < ids_b_to_a.size(); i++) {
      if (ids_b_to_a[i] >= 0) {
        assert(ids_b_to_a[i] < previous.spLandmarkNum() && "too large");
        auto prev_index = ids_b_to_a[i];
        auto landmark_id = previous.landmarks[prev_index].landmark_id;
        auto &cur_lm = frame.landmarks[i];
        auto [succ, prev_lm] = getPreviousLandmarkFrame(
            previous.landmarks[prev_index], base_kfframe.frame_id);
        if (!succ) {
          continue;
        }
        cur_lm.landmark_id = landmark_id;
        cur_lm.velocity = cur_lm.pt3d_norm - prev_lm.pt3d_norm;
        cur_lm.velocity /= (frame.stamp - base_frame.stamp);
        cur_lm.stamp_discover = prev_lm.stamp_discover;
        lmanager->updateLandmark(cur_lm);
        report.sum_parallex += (prev_lm.pt3d_norm - cur_lm.pt3d_norm).norm();
        // printf("[D2FeatureTracker::track] landmark %ld cam_idx %d<->%d
        // frame_cam_idx %d<->%d parallex %.1f%% prev_2d %.1f %.1f cur_2d %.3f
        // %.3f prev_3d %.3f %.3f %.3f cur_3d %.3f %.3f %.3f\n",
        //     landmark_id, prev_lm.camera_index, cur_lm.camera_index,
        //     previous.camera_index, frame.camera_index, (prev_lm.pt3d_norm -
        //     cur_lm.pt3d_norm).norm()*100, prev_lm.pt2d.x, prev_lm.pt2d.y,
        //     cur_lm.pt2d.x, cur_lm.pt2d.y, prev_lm.pt3d_norm.x(),
        //     prev_lm.pt3d_norm.y(), prev_lm.pt3d_norm.z(),
        //     cur_lm.pt3d_norm.x(), cur_lm.pt3d_norm.y(),
        //     cur_lm.pt3d_norm.z());
        report.parallex_num++;
        if (lmanager->at(landmark_id).track.size() >=
            _config.long_track_frames) {
          report.long_track_num++;
        } else {
          report.unmatched_num++;
        }
      }
    }
  }
  if (_config.enable_lk_optical_flow || _config.sp_track_use_lk) {
    // Enable LK optical flow feature tracker also.
    // This is for the case that the superpoint features is not tracked well.
    report.compose(trackLK(frame));
  }
  return report;
}

const VisualImageDescArray &D2FeatureTracker::getLatestKeyframe() const {
  // Return the previous keyframe
  assert(current_keyframes.size() > 0 && "Must have previous keyframe");
  for (auto it = current_keyframes.rbegin(); it != current_keyframes.rend();
       it++) {
    if (it->is_keyframe) {
      spdlog::debug("Found previous keyframe {}", it->frame_id);
      return *it;
    }
  }
  spdlog::debug("Not found previous keyframe {}, returning beginning");
  return *current_keyframes.begin();
}

TrackReport D2FeatureTracker::trackLK(VisualImageDesc &frame) {
  // Track LK points
  TrackReport report;
  LKImageInfoGPU cur_lk_info;
  auto cur_landmarks = frame.landmarks;
  auto cur_landmark_desc = frame.landmark_descriptor;
  auto cur_landmark_scores = frame.landmark_scores;
  if (_config.sp_track_use_lk) {
    frame.clearLandmarks();
  }

  if (_config.sp_track_use_lk || _config.lr_match_use_lk) {
    bool pyr_has_built = false;
    if (keyframe_lk_infos.size() > 0 && current_keyframes.size() > 0) {
      const auto &prev_frame = current_keyframes.back();
      const auto &prev_image = prev_frame.images[frame.camera_index];
      const auto &prev_keyframe = getLatestKeyframe();
      const auto &prev_lk =
          keyframe_lk_infos.at(prev_frame.frame_id).at(frame.camera_index);
      if (!prev_lk.lk_ids.empty()) {
        int prev_lk_num = prev_lk.lk_ids.size();
        cur_lk_info = opticalflowTrackPyr(frame.raw_image, prev_lk,
                                          TrackLRType::WHOLE_IMG_MATCH);
        pyr_has_built = true;
        cur_lk_info.lk_pts_3d_norm.resize(cur_lk_info.lk_pts.size());
        for (unsigned int i = 0; i < cur_lk_info.lk_pts.size(); i++) {
          auto ret =
              createLKLandmark(frame, cur_lk_info.lk_pts[i],
                               cur_lk_info.lk_ids[i], cur_lk_info.lk_types[i]);
          cur_lk_info.lk_pts_3d_norm[i] = ret.second.pt3d_norm;
          if (!ret.first) {
            continue;
          }
          if (_config.sp_track_use_lk) {
            // Copy the landmark descriptor from previous frame
            frame.landmark_descriptor.insert(
                frame.landmark_descriptor.end(),
                prev_image.landmark_descriptor.begin() +
                    cur_lk_info.lk_local_index[i] * params->superpoint_dims,
                prev_image.landmark_descriptor.begin() +
                    (cur_lk_info.lk_local_index[i] + 1) *
                        params->superpoint_dims);
            frame.landmark_scores.emplace_back(
                prev_image.landmark_scores[cur_lk_info.lk_local_index[i]]);
            cur_lk_info.lk_local_index[i] = frame.landmarks.size();
          }
          auto &lm = ret.second;
          auto track = lmanager->at(cur_lk_info.lk_ids[i]).track;
          lm.velocity = extractPointVelocity(lm);
          frame.landmarks.emplace_back(lm);
          auto [succ, prev_lm] =
              getPreviousLandmarkFrame(lm, prev_keyframe.frame_id);
          if (succ) {
            lm.stamp_discover = prev_lm.stamp_discover;
          } else {
            // SPDLOG_INFO("getPreviousLandmarkFrame failed");
            continue;
          }
          if (lmanager->at(cur_lk_info.lk_ids[i]).track.size() >=
              _config.long_track_frames) {
            report.long_track_num++;
          }

          report.sum_parallex += (lm.pt3d_norm - prev_lm.pt3d_norm).norm();
          // spdlog::debug("LM {} prev_2d {:.1f} {:.1f} cur_2d {:.3f} {:.3f}
          // para_2d {:.1f}%  prev_3d {:.3f} {:.3f} {:.3f} cur_3d {:.3f} {:.3f}
          // {:.3f} para_3d {:.1f}%",
          //         prev_lm.landmark_id, prev_lm.pt2d.x, prev_lm.pt2d.y,
          //         lm.pt2d.x, lm.pt2d.y, cv::norm(prev_lm.pt2d -
          //         lm.pt2d)*100.0, prev_lm.pt3d_norm.x(),
          //         prev_lm.pt3d_norm.y(), prev_lm.pt3d_norm.z(),
          //         lm.pt3d_norm.x(), lm.pt3d_norm.y(), lm.pt3d_norm.z(),
          //         (prev_lm.pt3d_norm - lm.pt3d_norm).norm()*100.0);
          report.parallex_num++;
        }
        // SPDLOG_INFO("[D2FeatureTracker::trackLK] track {} LK points, {} lost,
        // track rate {:.1f}% para {:.2f}% num {} {}->{}",
        //         prev_lk_num, prev_lk_num - cur_lk_info.lk_pts.size(),
        //         cur_lk_info.lk_pts.size() * 100.0 / prev_lk_num,
        //         report.meanParallex()*100, report.parallex_num,
        //         prev_frame.frame_id, frame.frame_id);
      }
    }
    if (!pyr_has_built) {
      cv::cuda::GpuMat image_cuda(frame.raw_image);
      cur_lk_info.pyr = buildImagePyramid(image_cuda);
    }
  }
  // Discover new points.
  if (!frame.raw_image.empty()) {
    if (_config.sp_track_use_lk) {
      // In this case, select from cur_landmarks
      int count_new = 0;
      for (size_t i = 0; i < cur_landmarks.size(); i++) {
        if (cur_lk_info.lk_pts.size() > params->total_feature_num) {
          break;
        }
        auto lm = cur_landmarks[i];
        // If not near to any existing landmarks, add it
        bool has_near = false;
        for (auto pt : cur_lk_info.lk_pts) {
          if (cv::norm(pt - lm.pt2d) < params->feature_min_dist) {
            has_near = true;
            break;
          }
        }
        if (!has_near) {
          auto _id = lmanager->addLandmark(lm);
          lm.landmark_id = _id;
          frame.landmarks.emplace_back(lm);
          frame.landmark_descriptor.insert(
              frame.landmark_descriptor.end(),
              cur_landmark_desc.begin() + i * params->superpoint_dims,
              cur_landmark_desc.begin() + (i + 1) * params->superpoint_dims);
          frame.landmark_scores.emplace_back(cur_landmark_scores[i]);

          cur_lk_info.lk_pts.emplace_back(lm.pt2d);
          cur_lk_info.lk_ids.emplace_back(lm.landmark_id);
          cur_lk_info.lk_local_index.emplace_back(frame.landmarks.size() - 1);
          cur_lk_info.lk_pts_3d_norm.emplace_back(lm.pt3d_norm);
          cur_lk_info.lk_types.emplace_back(LandmarkType::SuperPointLandmark);
          count_new++;
        }
      }
      spdlog::debug("{} new points added", cur_lk_info.lk_pts.size(),
                    count_new);
    } else {
      std::vector<cv::Point2f> n_pts;
      TicToc t_det;
      detectPoints(frame.raw_image, n_pts, frame.landmarks2D(),
                   params->total_feature_num, true, _config.lk_use_fast);
      spdlog::debug(
          "[D2FeatureTracker::trackLK] detect {} points in {:.2f}ms\n",
          n_pts.size(), t_det.toc());
      report.unmatched_num += n_pts.size();
      for (auto &pt : n_pts) {
        auto ret = createLKLandmark(frame, pt);
        if (!ret.first) {
          continue;
        }
        auto &lm = ret.second;
        auto _id = lmanager->addLandmark(lm);
        lm.landmark_id = _id;
        frame.landmarks.emplace_back(lm);
        cur_lk_info.lk_pts.emplace_back(pt);
        cur_lk_info.lk_ids.emplace_back(_id);
        cur_lk_info.lk_local_index.emplace_back(frame.landmarks.size() - 1);
        cur_lk_info.lk_pts_3d_norm.emplace_back(lm.pt3d_norm);
        cur_lk_info.lk_types.emplace_back(LandmarkType::FlowLandmark);
      }
    }
  } else {
    SPDLOG_ERROR("[D2FeatureTracker::trackLK] empty image\n");
  }

  keyframe_lk_infos[frame.frame_id][frame.camera_index] = cur_lk_info;
  return report;
}

std::pair<bool, LandmarkPerFrame> D2FeatureTracker::getPreviousLandmarkFrame(
    const LandmarkPerFrame &lpf, FrameIdType keyframe_id) const {
  auto landmark_id = lpf.landmark_id;
  // printf("[D2FeatureTracker::extractPointVelocity] landmark_id %d\n",
  // landmark_id);
  if (lmanager->hasLandmark(landmark_id) &&
      lmanager->at(landmark_id).track.size() > 0) {
    auto lm_per_id = lmanager->at(landmark_id);
    for (int i = lm_per_id.track.size() - 1; i >= 0; i--) {
      auto lm = lm_per_id.track[i];
      if (lm.landmark_id == landmark_id && lm.camera_id == lpf.camera_id &&
          (keyframe_id < 0 || lm.frame_id == keyframe_id)) {
        return {true, lm};
      }
    }
  }
  LandmarkPerFrame lm;
  return {false, lm};
}

Vector3d D2FeatureTracker::extractPointVelocity(
    const LandmarkPerFrame &lpf) const {
  auto landmark_id = lpf.landmark_id;
  // printf("[D2FeatureTracker::extractPointVelocity] landmark_id %d\n",
  // landmark_id);
  auto [succ, lm] = getPreviousLandmarkFrame(lpf);
  if (succ) {
    Vector3d movement = lpf.pt3d_norm - lm.pt3d_norm;
    auto vel = movement / (lpf.stamp - lm.stamp);
    // printf("[D2FeatureTracker::extractPointVelocity] landmark %d, frame
    // %d->%d, movement %f %f %f vel  %f %f %f \n",
    //     landmark_id, lm.frame_id, lpf.frame_id, movement.x(), movement.y(),
    //     movement.z(), vel.x(), vel.y(), vel.z());
    return vel;
  }
  return Vector3d(0, 0, 0);
}

TrackReport D2FeatureTracker::track(const VisualImageDesc &left_frame,
                                    VisualImageDesc &right_frame,
                                    bool enable_lk, TrackLRType type,
                                    bool use_lk_for_left_right_track) {
  auto prev_pts = left_frame.landmarks2D();
  auto cur_pts = right_frame.landmarks2D();
  std::vector<int> ids_b_to_a;
  TrackReport report;
  if (!use_lk_for_left_right_track) {
    MatchLocalFeatureParams match_param;
    match_param.enable_superglue = _config.enable_superglue_local;
    match_param.search_radius = _config.search_local_max_dist_lr * image_width;
    match_param.type = type;
    match_param.enable_prediction = true;
    match_param.prediction_using_extrinsic = true;
    match_param.enable_search_in_local = true;
    matchLocalFeatures(left_frame, right_frame, ids_b_to_a, match_param);
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
        report.stereo_point_num++;
      }
    }
  }
  if (_config.enable_lk_optical_flow && enable_lk ||
      use_lk_for_left_right_track) {
    trackLK(left_frame, right_frame, type, use_lk_for_left_right_track);
  }
  return report;
}

TrackReport D2FeatureTracker::trackLK(const VisualImageDesc &left_frame,
                                      VisualImageDesc &right_frame,
                                      TrackLRType type,
                                      bool use_lk_for_left_right_track) {
  // Track LK points
  // This function MUST run after track(...)
  TrackReport report;
  auto left_lk_info =
      keyframe_lk_infos.at(left_frame.frame_id).at(left_frame.camera_index);
  // Add the SP points to the LK points if use_lk_for_left_right_track is true
  if (use_lk_for_left_right_track && !_config.sp_track_use_lk) {
    for (unsigned int i = 0; i < left_frame.landmarkNum(); i++) {
      if (left_frame.landmarks[i].landmark_id >= 0 &&
          left_frame.landmarks[i].type == LandmarkType::SuperPointLandmark) {
        left_lk_info.lk_pts.emplace_back(left_frame.landmarks[i].pt2d);
        left_lk_info.lk_pts_3d_norm.emplace_back(
            left_frame.landmarks[i].pt3d_norm);
        left_lk_info.lk_ids.emplace_back(left_frame.landmarks[i].landmark_id);
        left_lk_info.lk_local_index.emplace_back(i);
        left_lk_info.lk_types.emplace_back(left_frame.landmarks[i].type);
      }
    }
  }
  std::map<LandmarkIdType, cv::Point2f> pts_pred_a_on_b;
  if (_config.lk_lk_use_pred) {
    pts_pred_a_on_b = predictLandmarksWithExtrinsic(
        left_frame.camera_index, left_lk_info.lk_ids,
        left_lk_info.lk_pts_3d_norm, left_frame.extrinsic,
        right_frame.extrinsic);
  }
  if (!left_lk_info.lk_ids.empty()) {
    auto cur_lk_info =
        opticalflowTrackPyr(right_frame.raw_image, left_lk_info, type);
    for (unsigned int i = 0; i < cur_lk_info.lk_pts.size(); i++) {
      auto ret =
          createLKLandmark(right_frame, cur_lk_info.lk_pts[i],
                           cur_lk_info.lk_ids[i], cur_lk_info.lk_types[i]);
      if (!ret.first) {
        continue;
      }
      auto &lm = ret.second;
      lm.stamp_discover = lmanager->at(cur_lk_info.lk_ids[i]).stamp_discover;
      lm.velocity = extractPointVelocity(lm);
      lmanager->updateLandmark(lm);
      auto pred = pts_pred_a_on_b.at(cur_lk_info.lk_ids[i]);
      if (!_config.lk_lk_use_pred ||
          cv::norm(pred - cur_lk_info.lk_pts[i]) <
              _config.search_local_max_dist_lr * image_width) {
        right_frame.landmarks.emplace_back(lm);
      }
    }

    report.stereo_point_num = right_frame.landmarks.size();
  }
  return report;
}

bool D2FeatureTracker::isKeyframe(const TrackReport &report) {
  int prev_num =
      current_keyframes.size() > 0 ? current_keyframes.back().landmarkNum() : 0;
  if (report.meanParallex() > 0.5) {
    printf("unexcepted mean parallex %f\n", report.meanParallex());
  }
  if (report.parallex_num < _config.min_keyframe_num ||
      report.long_track_num < _config.long_track_thres ||
      prev_num < _config.last_track_thres ||
      report.unmatched_num > _config.new_feature_thres *
                                 prev_num ||  // Unmatched is assumed to be new
      report.meanParallex() >
          _config.parallex_thres) {  // Attenion, if mismatch this will be big
    spdlog::debug(
        "New KF: keyframe_count: {}, long_track_num: {}, prev_num: {}, "
        "unmatched_num: {}, parallex: {:.1f}%",
        keyframe_count, report.long_track_num, prev_num, report.unmatched_num,
        report.meanParallex() * 100);
    return true;
  }
  return false;
}

std::pair<bool, LandmarkPerFrame> D2FeatureTracker::createLKLandmark(
    const VisualImageDesc &frame, cv::Point2f pt, LandmarkIdType landmark_id,
    LandmarkType type) {
  Vector3d pt3d_norm = Vector3d::Zero();
  cams.at(frame.camera_index)
      ->liftProjective(Eigen::Vector2d(pt.x, pt.y), pt3d_norm);
  pt3d_norm.normalize();
  if (pt3d_norm.hasNaN()) {
    return std::make_pair(false, LandmarkPerFrame());
  }
  LandmarkPerFrame lm = LandmarkPerFrame::createLandmarkPerFrame(
      landmark_id, frame.frame_id, frame.stamp, type, params->self_id,
      frame.camera_index, frame.camera_id, pt, pt3d_norm);
  if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
    // Add depth information
    auto dep = frame.raw_depth_image.at<unsigned short>(pt) / 1000.0;
    if (dep > params->loopcamconfig->DEPTH_NEAR_THRES &&
        dep < params->loopcamconfig->DEPTH_FAR_THRES) {
      auto pt3dcam = pt3d_norm * dep;
      lm.depth = pt3dcam.norm();
      lm.depth_mea = true;
    }
  }
  lm.color = extractColor(frame.raw_image, pt);
  return std::make_pair(true, lm);
}

void D2FeatureTracker::processFrame(VisualImageDescArray &frames,
                                    bool is_keyframe) {
  if (current_keyframes.size() > 0 &&
      current_keyframes.back().frame_id == frames.frame_id) {
    return;
  }
  frames.is_keyframe = is_keyframe;
  keyframe_count++;
  for (auto &frame : frames.images) {
    for (unsigned int i = 0; i < frame.landmarkNum(); i++) {
      if (frame.landmarks[i].landmark_id < 0) {
        if (params->camera_configuration == CameraConfig::STEREO_PINHOLE &&
            frame.camera_index == 1) {
          // We do not create new landmark for right camera
          continue;
        }
        auto _id = lmanager->addLandmark(frame.landmarks[i]);
        frame.landmarks[i].setLandmarkId(_id);
      } else {
        lmanager->updateLandmark(frame.landmarks[i]);
      }
    }
  }
  // Before solve, use motion prediction as pose
  for (auto &frame : frames.images) {
    frame.pose_drone = frames.motion_prediction;
  }
  frames.pose_drone = frames.motion_prediction;
  if (is_keyframe || !_config.track_from_keyframe) {
    spdlog::debug(
        "[D2FeatureTracker::processFrame] Add to keyframe list, frame_id: {}",
        frames.frame_id);
    if (current_keyframes.size() > 0) {
      keyframe_lk_infos.erase(current_keyframes.back().frame_id);
    }
    current_keyframes.emplace_back(frames);
  } else {
    keyframe_lk_infos.erase(frames.frame_id);
  }
}

cv::Mat D2FeatureTracker::drawToImage(const VisualImageDesc &frame,
                                      bool is_keyframe,
                                      const TrackReport &report, bool is_right,
                                      bool is_remote) const {
  // ROS_INFO("Drawing ... %d", keyframe_count);
  cv::Mat img = frame.raw_image;
  int width = img.cols;
  auto &current_keyframe = current_keyframes.back();
  FrameIdType last_keyframe = -1;
  if (current_keyframes.size() > 1) {
    last_keyframe = current_keyframes[current_keyframes.size() - 2].frame_id;
  }
  if (is_remote) {
    img = cv::imdecode(frame.image, cv::IMREAD_UNCHANGED);
    width = img.cols;
    if (img.empty()) {
      return cv::Mat();
    }
    // cv::hconcat(img, current_keyframe.images[0].raw_image, img);
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
      color = cv::Scalar(255, 0, 0);  // Superpoint blue
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
      auto &pts2d = lmanager->at(_id).track;
      if (pts2d.size() == 0) continue;
      if (is_remote) {
        prev = pts2d.back().pt2d;
        prev_found = true;
      } else {
        for (int index = pts2d.size() - 1; index >= 0; index--) {
          if (!is_right && pts2d[index].camera_id == frame.camera_id &&
              pts2d[index].frame_id == last_keyframe) {
            prev = lmanager->at(_id).track[index].pt2d;
            prev_found = true;
            break;
          }

          if (is_right && pts2d[index].frame_id == frame.frame_id &&
              pts2d[index].camera_id != frame.camera_id) {
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
        // Random color
        // cv::Scalar color(rand()%255, rand()%255, rand()%255);
        // cv::line(img, prev + cv::Point2f(width, 0), cur_pts[j], color);
      } else {
        cv::arrowedLine(img, prev, cur_pts[j], cv::Scalar(0, 255, 0), 1, 8, 0,
                        0.2);
      }
    }
    if (frame.landmarks[j].landmark_id >= 0) {
      stereo_num++;
    }
    if (_config.show_feature_id && frame.landmarks[j].landmark_id >= 0) {
      sprintf(buf, "%d", frame.landmarks[j].landmark_id % MAX_FEATURE_NUM);
      cv::putText(img, buf, cur_pts[j] - cv::Point2f(5, 0),
                  cv::FONT_HERSHEY_SIMPLEX, 1, color, 1);
    }
  }
  // Draw predictions
  if (landmark_predictions_viz.find(frame.camera_id) !=
      landmark_predictions_viz.end()) {
    // Draw predictions here
    auto &predictions = landmark_predictions_viz.at(frame.camera_id);
    auto &prev = landmark_predictions_matched_viz.at(frame.camera_id);
    for (unsigned int i = 0; i < predictions.size(); i++) {
      cv::circle(img, predictions[i], 3, cv::Scalar(0, 255, 0), 2);
      cv::line(img, prev[i], predictions[i], cv::Scalar(0, 0, 255), 1, 8, 0);
      // if (cv::norm(prev[i] - predictions[i]) > 20) {
      //     //Show the landmark id and flag
      //     auto lm_id = frame.landmarks[i].landmark_id;
      //     if (lmanager->hasLandmark(lm_id)) {
      //         printf("Landmark %ld: flag %d %.2f %.2f err %.2f\n",
      //         frame.landmarks[i].landmark_id,
      //             lmanager->at(lm_id).flag, prev[i].x, prev[i].y,
      //             cv::norm(prev[i] - predictions[i]));
      //     }
      // }
    }
  }
  cv::Scalar color = cv::Scalar(255, 0, 0);
  if (is_keyframe) {
    color = cv::Scalar(0, 0, 255);
  }
  if (is_right) {
    sprintf(buf, "Stereo points: %d", stereo_num);
    cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2);
  } else if (is_remote) {
    sprintf(buf, "Drone %d<->%d Matched points: %d", params->self_id,
            frame.drone_id, report.remote_matched_num);
    cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2);
  } else {
    sprintf(buf, "KF/FRAME %d/%d @CAM %d ISKF: %d", keyframe_count, frame_count,
            frame.camera_index, is_keyframe);
    cv::putText(img, buf, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2);
    sprintf(buf, "TRACK %.1fms NUM %d LONG %d Parallex %.1f\%/%.1f",
            report.ft_time, report.parallex_num, report.long_track_num,
            report.meanParallex() * 100, _config.parallex_thres * 100);
    cv::putText(img, buf, cv::Point2f(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2);
  }
  return img;
}

void D2FeatureTracker::drawRemote(const VisualImageDescArray &frames,
                                  const TrackReport &report) const {
  cv::Mat img = drawToImage(frames.images[0], false, report, false, true);
  cv::Mat img_r = drawToImage(frames.images[1], false, report, true, true);
  if (img.empty()) {
    printf(
        "[D2FeatureTracker::drawRemote] Unable to draw remote image, empty "
        "image found\n");
    return;
  }
  char buf[64] = {0};
  sprintf(buf, "RemoteMatched @ Drone %d", params->self_id);
  cv::hconcat(img, img_r, img);
  cv::imshow(buf, img);
  cv::waitKey(1);
  if (_config.write_to_file) {
    sprintf(buf, "%s/featureTracker_remote%06d.jpg",
            _config.output_folder.c_str(), frame_count);
    cv::imwrite(buf, img);
  }
}

void D2FeatureTracker::draw(const VisualImageDesc &frame, bool is_keyframe,
                            const TrackReport &report) const {
  cv::Mat img = drawToImage(frame, is_keyframe, report);
  char buf[64] = {0};
  sprintf(buf, "featureTracker @ Drone %d", params->self_id);
  cv::imshow(buf, img);
  cv::waitKey(1);
  if (_config.write_to_file) {
    sprintf(buf, "%s/featureTracker%06d.jpg", _config.output_folder.c_str(),
            frame_count);
    cv::imwrite(buf, img);
  }
}

void D2FeatureTracker::draw(const VisualImageDesc &lframe,
                            VisualImageDesc &rframe, bool is_keyframe,
                            const TrackReport &report) const {
  cv::Mat img = drawToImage(lframe, is_keyframe, report);
  cv::Mat img_r = drawToImage(rframe, is_keyframe, report, true);
  cv::hconcat(img, img_r, img);
  char buf[64] = {0};
  sprintf(buf, "featureTracker @ Drone %d", params->self_id);
  cv::imshow(buf, img);
  cv::waitKey(1);
  if (_config.write_to_file) {
    sprintf(buf, "%s/featureTracker%06d.jpg", _config.output_folder.c_str(),
            frame_count);
    cv::imwrite(buf, img);
  }
}

void D2FeatureTracker::draw(const VisualImageDescArray &frames,
                            bool is_keyframe, const TrackReport &report) const {
  cv::Mat img = drawToImage(frames.images[0], is_keyframe, report);
  cv::Mat img_r = drawToImage(frames.images[2], is_keyframe, report);
  cv::hconcat(img, img_r, img);
  cv::Mat img1 = drawToImage(frames.images[1], is_keyframe, report);
  cv::Mat img1_r = drawToImage(frames.images[3], is_keyframe, report);
  cv::hconcat(img1, img1_r, img1);
  cv::vconcat(img, img1, img);

  char buf[64] = {0};
  sprintf(buf, "featureTracker @ Drone %d", params->self_id);
  cv::imshow(buf, img);
  cv::waitKey(1);
  if (_config.write_to_file) {
    sprintf(buf, "%s/featureTracker%06d.jpg", _config.output_folder.c_str(),
            frame_count);
    cv::imwrite(buf, img);
  }
}

std::pair<std::vector<float>, std::vector<cv::Point2f>> getFeatureHalfImg(
    const std::vector<cv::Point2f> &pts, const std::vector<float> &desc,
    bool require_left, std::map<int, int> &tmp_to_idx) {
  std::vector<float> desc_half;
  std::vector<cv::Point2f> pts_new;
  desc_half.resize(desc.size());
  int c = 0;
  float move_cols =
      params->width_undistort * 90.0 /
      params->undistort_fov;  // slightly lower than 0.5 cols when fov=200
  for (unsigned int i = 0; i < pts.size(); i++) {
    if (require_left && pts[i].x < params->width_undistort - move_cols ||
        !require_left && pts[i].x >= move_cols) {
      tmp_to_idx[c] = i;
      // Copy from desc to desc_half
      std::copy(desc.begin() + i * params->superpoint_dims,
                desc.begin() + (i + 1) * params->superpoint_dims,
                desc_half.begin() + c * params->superpoint_dims);
      pts_new.push_back(pts[i]);
      c += 1;
    }
  }
  desc_half.resize(c * params->superpoint_dims);
  return std::make_pair(desc_half, pts_new);
}

bool D2FeatureTracker::matchLocalFeatures(
    const VisualImageDesc &img_desc_a, const VisualImageDesc &img_desc_b,
    std::vector<int> &ids_b_to_a,
    const D2FeatureTracker::MatchLocalFeatureParams &param) {
  TicToc tic;
  auto &raw_desc_a = img_desc_a.landmark_descriptor;
  auto &raw_desc_b = img_desc_b.landmark_descriptor;
  auto pts_a = img_desc_a.landmarks2D(true);
  auto pts_b = img_desc_b.landmarks2D(true);
  auto pts_a_normed = img_desc_a.landmarks2D(true, true);
  auto pts_b_normed = img_desc_b.landmarks2D(true, true);
  std::vector<int> ids_a, ids_b;
  std::vector<cv::DMatch> _matches;
  std::vector<cv::Point2f> pts_pred_a_on_b;
  ids_b_to_a.resize(pts_b.size());
  std::fill(ids_b_to_a.begin(), ids_b_to_a.end(), -1);
  double search_radius = param.search_radius;
  spdlog::debug(
      "Match {}<->{} enable_prediction {} pose_a {} enable_search_in_local {} "
      "motion_prediction {} prediction_using_extrinsic {}",
      img_desc_a.frame_id, img_desc_b.frame_id, param.enable_prediction,
      param.pose_a.toStr(), param.enable_search_in_local,
      param.pose_b_prediction.toStr(), param.prediction_using_extrinsic);
  if (param.enable_prediction) {
    if (param.prediction_using_extrinsic) {
      pts_pred_a_on_b = predictLandmarks(img_desc_a, img_desc_a.extrinsic,
                                         img_desc_b.extrinsic, true);
    } else {
      pts_pred_a_on_b =
          predictLandmarks(img_desc_a, param.pose_a * img_desc_a.extrinsic,
                           param.pose_b_prediction * img_desc_b.extrinsic);
    }

  } else {
    pts_pred_a_on_b = pts_a;
    if (!param.enable_search_in_local) {
      search_radius = -1;
    }
  }
  if (param.enable_superglue) {
    // Superglue only support whole image matching
    auto &scores0 = img_desc_a.landmark_scores;
    auto &scores1 = img_desc_b.landmark_scores;
    _matches = superglue->inference(pts_a, pts_b, raw_desc_a, raw_desc_b,
                                    scores0, scores1);
  } else {
    if (param.type == WHOLE_IMG_MATCH) {
      const cv::Mat desc_a(raw_desc_a.size() / params->superpoint_dims,
                           params->superpoint_dims, CV_32F,
                           const_cast<float *>(raw_desc_a.data()));
      const cv::Mat desc_b(raw_desc_b.size() / params->superpoint_dims,
                           params->superpoint_dims, CV_32F,
                           const_cast<float *>(raw_desc_b.data()));
      if (_config.enable_knn_match) {
        if (img_desc_a.drone_id == img_desc_b.drone_id &&
            img_desc_a.camera_id == img_desc_b.camera_id) {
          // Is continuous frame
          _matches = matchKNN(desc_a, desc_b, _config.knn_match_ratio,
                              pts_pred_a_on_b, pts_b, search_radius);
        } else {
          _matches = matchKNN(desc_a, desc_b, _config.knn_match_ratio,
                              pts_pred_a_on_b, pts_b, search_radius);
        }
      } else {
        cv::BFMatcher bfmatcher(cv::NORM_L2, true);
        bfmatcher.match(desc_a, desc_b, _matches);  // Query train result
      }
    } else {
      // TODO: motion prediction for quadcam on stereo
      std::map<int, int> tmp_to_idx_a, tmp_to_idx_b;
      auto features_a = getFeatureHalfImg(
          pts_a, raw_desc_a, param.type == LEFT_RIGHT_IMG_MATCH, tmp_to_idx_a);
      auto features_b = getFeatureHalfImg(
          pts_b, raw_desc_b, param.type == RIGHT_LEFT_IMG_MATCH, tmp_to_idx_b);
      if (tmp_to_idx_a.size() == 0 || tmp_to_idx_b.size() == 0) {
        spdlog::debug("matchLocalFeatures failed: no feature to match.\n");
        return false;
      }
      cv::BFMatcher bfmatcher(cv::NORM_L2, true);
      const cv::Mat desc_a(tmp_to_idx_a.size(), params->superpoint_dims, CV_32F,
                           const_cast<float *>(features_a.first.data()));
      const cv::Mat desc_b(tmp_to_idx_b.size(), params->superpoint_dims, CV_32F,
                           const_cast<float *>(features_b.first.data()));
      if (_config.enable_knn_match) {
        if (param.enable_search_in_local) {
          float move_cols =
              params->width_undistort * 90.0 /
              params
                  ->undistort_fov;  // slightly lower than 0.5 cols when fov=200
          for (unsigned int i = 0; i < features_a.second.size(); i++) {
            features_a.second[i].x +=
                param.type == LEFT_RIGHT_IMG_MATCH ? move_cols : -move_cols;
          }
        }
        _matches =
            matchKNN(desc_a, desc_b, _config.knn_match_ratio, features_a.second,
                     features_b.second, search_radius);
      } else {
        cv::BFMatcher bfmatcher(cv::NORM_L2, true);
        bfmatcher.match(desc_a, desc_b, _matches);
      }
      for (auto &match : _matches) {
        match.queryIdx = tmp_to_idx_a[match.queryIdx];
        match.trainIdx = tmp_to_idx_b[match.trainIdx];
      }
    }
  }
  std::vector<cv::Point2f> matched_pts_a_normed, matched_pts_b_normed,
      matched_pts_a, matched_pts_b;
  if (params->show) {
    landmark_predictions_viz[img_desc_b.camera_id] = std::vector<cv::Point2f>();
    landmark_predictions_matched_viz[img_desc_b.camera_id] =
        std::vector<cv::Point2f>();
  }
  for (auto match : _matches) {
    ids_a.push_back(match.queryIdx);
    ids_b.push_back(match.trainIdx);
    matched_pts_a_normed.push_back(pts_a_normed[match.queryIdx]);
    matched_pts_b_normed.push_back(pts_b_normed[match.trainIdx]);
    matched_pts_a.push_back(pts_a[match.queryIdx]);
    matched_pts_b.push_back(pts_b[match.trainIdx]);
    if (params->show && param.enable_prediction) {
      landmark_predictions_viz[img_desc_b.camera_id].push_back(
          pts_pred_a_on_b[match.queryIdx]);
      landmark_predictions_matched_viz[img_desc_b.camera_id].push_back(
          pts_b[match.trainIdx]);
      // printf("Point %d: (%f, %f) -> (%f, %f)\n", match.queryIdx,
      //     pts_pred_a_on_b[match.queryIdx].x,
      //     pts_pred_a_on_b[match.queryIdx].y, pts_b[match.trainIdx].x,
      //     pts_b[match.trainIdx].y);
    }
  }
  if (img_desc_a.drone_id != img_desc_b.drone_id && _config.check_essential &&
      !param.enable_superglue) {
    // only perform this for remote
    std::vector<unsigned char> mask;
    if (matched_pts_a_normed.size() < MIN_HOMOGRAPHY) {
      spdlog::debug(
          "matchLocalFeatures failed only %ld pts not meet MIN_HOMOGRAPHY",
          matched_pts_a_normed.size());
      return false;
    }
    // cv::findHomography(matched_pts_a, matched_pts_b, cv::RANSAC,
    // _config.ransacReprojThreshold, mask); Find essential matrix with
    // normalized points
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0.0, 0, 1.0, 0.0, 0, 0, 1);
    cv::findEssentialMat(
        matched_pts_a_normed, matched_pts_b_normed, K, cv::RANSAC, 0.999,
        _config.ransacReprojThreshold / params->focal_length, mask);
    reduceVector(ids_a, mask);
    reduceVector(ids_b, mask);
    reduceVector(matched_pts_a, mask);
    reduceVector(matched_pts_b, mask);
  }
  for (auto i = 0; i < ids_a.size(); i++) {
    if (ids_a[i] >= pts_a.size()) {
      SPDLOG_ERROR("ids_a[i] > pts_a.size() why is this case?");
      continue;
    }
    ids_b_to_a[ids_b[i]] = ids_a[i];
  }

  // //Plot matches
  if (param.plot) {
    char name[100];
    std::vector<cv::KeyPoint> kps_a, kps_b;
    // Kps from points
    for (unsigned int i = 0; i < pts_a.size(); i++) {
      kps_a.push_back(cv::KeyPoint(pts_a[i].x, pts_a[i].y, 1));
    }
    for (unsigned int i = 0; i < pts_b.size(); i++) {
      kps_b.push_back(cv::KeyPoint(pts_b[i].x, pts_b[i].y, 1));
    }
    cv::Mat show;
    cv::Mat image_b = img_desc_b.raw_image;
    if (image_b.empty()) {
      cv::imdecode(img_desc_b.image, cv::IMREAD_GRAYSCALE, &image_b);
    }
    cv::drawMatches(img_desc_a.raw_image, kps_a, image_b, kps_b, _matches,
                    show);
    sprintf(name, "Matched points: %d", _matches.size());
    cv::putText(show, name, cv::Point2f(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
    sprintf(name, "matches_CAM@Drone %d@%d_%d@%d", img_desc_a.camera_index,
            img_desc_a.drone_id, img_desc_b.camera_index, img_desc_b.drone_id);
    if (_config.check_essential) {
      cv::Mat show_check;
      cv::hconcat(img_desc_a.raw_image, image_b, show_check);
      cv::cvtColor(show_check, show_check, cv::COLOR_GRAY2BGR);
      for (unsigned int i = 0; i < matched_pts_a.size(); i++) {
        // random color
        cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
        cv::line(show_check, matched_pts_a[i],
                 matched_pts_b[i] + cv::Point2f(show.cols / 2, 0), color, 1);
      }
      sprintf(name, "filtered_matches_CAM@Drone %d@%d_%d@%d",
              img_desc_a.camera_index, img_desc_a.drone_id,
              img_desc_b.camera_index, img_desc_b.drone_id);
      cv::vconcat(show, show_check, show);
      cv::imshow(name, show);
    } else {
      cv::imshow(name, show);
    }
  }

  spdlog::debug(
      "match features {}:{} matched inliers{}/all{} frame {}:{} t: {:.3f}ms "
      "enable_knn {} kNN ratio {} search_dist {:.2f} check_essential {} "
      "sp_dims {}",
      pts_a.size(), pts_b.size(), ids_b.size(), _matches.size(),
      img_desc_a.frame_id, img_desc_b.frame_id, tic.toc(),
      _config.enable_knn_match, _config.knn_match_ratio, search_radius,
      _config.check_essential, params->superpoint_dims);
  if (ids_b.size() >= _config.remote_min_match_num) {
    return true;
  }
  return false;
}

std::map<LandmarkIdType, cv::Point2f>
D2FeatureTracker::predictLandmarksWithExtrinsic(
    int camera_index, std::vector<LandmarkIdType> pts_ids,
    std::vector<Eigen::Vector3d> pts_3d_norm, const Swarm::Pose &cam_pose_a,
    const Swarm::Pose &cam_pose_b) const {
  std::map<LandmarkIdType, cv::Point2f> pts_a_pred_on_b;
  auto cam = cams.at(camera_index);
  for (unsigned int i = 0; i < pts_3d_norm.size(); i++) {
    Vector3d landmark_pos_cam =
        pts_3d_norm[i] * _config.landmark_distance_assumption;
    Vector3d pt3d = cam_pose_a * landmark_pos_cam;
    Vector2d pt2d_pred;
    Vector3d pos_cam_b_pred = cam_pose_b.inverse() * pt3d;
    cam->spaceToPlane(pos_cam_b_pred, pt2d_pred);
    pts_a_pred_on_b[pts_ids[i]] = {pt2d_pred.x(), pt2d_pred.y()};
  }
  return pts_a_pred_on_b;
}

std::vector<cv::Point2f> D2FeatureTracker::predictLandmarks(
    const VisualImageDesc &img_desc_a, const Swarm::Pose &cam_pose_a,
    const Swarm::Pose &cam_pose_b, bool use_extrinsic) const {
  std::vector<cv::Point2f> pts_a_pred_on_b;
  assert(img_desc_a.drone_id == params->self_id);
  auto cam = cams.at(img_desc_a.camera_index);
  for (unsigned int i = 0; i < img_desc_a.spLandmarkNum(); i++) {
    auto landmark_id = img_desc_a.landmarks[i].landmark_id;
    // Query 3d landmark position
    bool find_position = false;
    Vector3d pt3d(0., 0., 0.);
    if (!use_extrinsic && lmanager->hasLandmark(landmark_id)) {
      const auto &lm = lmanager->at(landmark_id);
      if (lm.flag == LandmarkFlag::INITIALIZED ||
          lm.flag == LandmarkFlag::ESTIMATED) {
        pt3d = lm.position;
        find_position = true;
      }
    }
    if (!find_position) {
      Vector3d landmark_pos_cam = img_desc_a.landmarks[i].pt3d_norm *
                                  _config.landmark_distance_assumption;
      pt3d = cam_pose_a * landmark_pos_cam;
    }
    // Predict 2d position on b
    Vector2d pt2d_pred;
    Vector3d pos_cam_b_pred = cam_pose_b.inverse() * pt3d;
    cam->spaceToPlane(pos_cam_b_pred, pt2d_pred);
    pts_a_pred_on_b.emplace_back(pt2d_pred.x(), pt2d_pred.y());
    // printf("[D2FT] Frame_a %ld landmark %ld find_pos:%d position %.2f %.2f
    // %.2f pred %.2f %.2f\n",
    //     img_desc_a.frame_id, landmark_id, find_position, pt3d.x(), pt3d.y(),
    //     pt3d.z(), pt2d_pred.x(), pt2d_pred.y());
  }
  return pts_a_pred_on_b;
}

}  // namespace D2FrontEnd
