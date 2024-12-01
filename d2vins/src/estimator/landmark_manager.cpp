#include "landmark_manager.hpp"

#include <opencv2/core/eigen.hpp>

#include "../d2vins_params.hpp"
#include "../factors/reprojection3d.h"
#include "../utils/solve_5pts.h"
#include "d2frontend/utils.h"
#include "d2vinsstate.hpp"
#include "spdlog/spdlog.h"

namespace D2VINS {

double triangulatePoint3DPts(const std::vector<Swarm::Pose> poses,
                             const std::vector<Vector3d> &points,
                             Vector3d &point_3d);

void D2LandmarkManager::addKeyframe(const VisualImageDescArray &images,
                                    double td) {
  const Guard lock(state_lock);
  for (auto &image : images.images) {
    for (auto lm : image.landmarks) {
      if (lm.landmark_id < 0) {
        // Do not use unmatched features.
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

std::vector<LandmarkPerId> D2LandmarkManager::availableMeasurements(
    int max_pts, int max_solve_measurements,
    const std::set<FrameIdType> &current_frames) const {
  const Guard lock(state_lock);
  std::map<FrameIdType, int> current_landmark_num;
  std::map<FrameIdType, int> result_landmark_num;
  std::map<FrameIdType, std::set<D2Common::LandmarkIdType>>
      current_assoicated_landmarks;
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
    // found the frame with minimum landmarks in current frames
    if (current_landmark_num.size() == 0) {
      exit = true;
    }
    auto it =
        min_element(current_landmark_num.begin(), current_landmark_num.end(),
                    [](decltype(current_landmark_num)::value_type &l,
                       decltype(current_landmark_num)::value_type &r) -> bool {
                      return l.second < r.second;
                    });
    auto frame_id = it->first;
    // Add the a landmark in its related landmarks with highest score
    if (related_landmarks.find(frame_id) == related_landmarks.end()) {
      // Remove the frame from current_landmark_num
      current_landmark_num.erase(frame_id);
      continue;
    }
    auto frame_related_landmarks = related_landmarks.at(frame_id);
    // Find the landmark with highest score
    LandmarkIdType lm_best;
    double score_best = -10000;
    bool found = false;
    for (auto &itre : frame_related_landmarks) {
      LandmarkIdType lm_id = itre.first;
      if (landmark_db.find(lm_id) == landmark_db.end() ||
          ret_ids_set.find(lm_id) != ret_ids_set.end()) {
        // The landmark is not in the database or has been added
        continue;
      }
      auto &lm = landmark_db.at(lm_id);
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
      auto &lm = landmark_db.at(lm_best);
      ret_set.emplace_back(lm);
      ret_ids_set.insert(lm_best);
      count_measurements += lm.track.size();
      // Add the frame to current_landmark_num
      for (auto track : lm.track) {
        auto frame_id = track.frame_id;
        current_assoicated_landmarks[frame_id].insert(lm_best);
        // We count the landmark numbers, but not the measurements
        current_landmark_num[frame_id] =
            current_assoicated_landmarks[frame_id].size();
        result_landmark_num[frame_id] = current_landmark_num[frame_id];
      }
      if (ret_set.size() >= max_pts ||
          count_measurements >= max_solve_measurements) {
        exit = true;
      }
    } else {
      // Remove the frame from current_landmark_num
      current_landmark_num.erase(frame_id);
    }
  }
  if (params->verbose) {
    printf(
        "[D2VINS::D2LandmarkManager] Found %ld(total %ld) landmarks "
        "measure %d/%d in %ld frames\n",
        ret_set.size(), landmark_db.size(), count_measurements,
        max_solve_measurements, result_landmark_num.size());
  }
  return ret_set;
}

double *D2LandmarkManager::getLandmarkState(LandmarkIdType landmark_id) const {
  const Guard lock(state_lock);
  return landmark_state.at(landmark_id);
}

void D2LandmarkManager::moveByPose(const Swarm::Pose &delta_pose) {
  const Guard lock(state_lock);
  for (auto it : landmark_db) {
    auto &lm = it.second;
    if (lm.flag != LandmarkFlag::UNINITIALIZED) {
      lm.position = delta_pose * lm.position;
    }
  }
}

void D2LandmarkManager::initialLandmarkState(LandmarkPerId &lm,
                                             const D2EstimatorState *state) {
  const Guard lock(state_lock);
  LandmarkPerFrame lm_first;
  lm_first = lm.track[0];
  auto lm_id = lm.landmark_id;
  auto pt3d_n = lm_first.pt3d_norm;
  auto firstFrame = *state->getFramebyId(lm_first.frame_id);
  // printf("[D2VINS::D2LandmarkManager] Try initial landmark %ld dep %d
  // tracks %ld\n", lm_id,
  //     lm.track[0].depth_mea && lm.track[0].depth >
  //     params->min_depth_to_fuse && lm.track[0].depth <
  //     params->max_depth_to_fuse, lm.track.size());
  if (lm_first.depth_mea && lm_first.depth > params->min_depth_to_fuse &&
      lm_first.depth < params->max_depth_to_fuse) {
    // Use depth to initial
    auto ext = state->getExtrinsic(lm_first.camera_id);
    // Note in depth mode, pt3d = (u, v, w), depth is distance since we use
    // unitsphere
    Vector3d pos = pt3d_n * lm_first.depth;
    pos = firstFrame.odom.pose() * ext * pos;
    lm.position = pos;
    if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
      *landmark_state[lm_id] = 1 / lm_first.depth;
      if (params->debug_print_states) {
        printf(
            "[D2VINS::D2LandmarkManager] Initialize landmark %ld by "
            "depth measurement position %.3f %.3f %.3f inv_dep %.3f\n",
            lm_id, pos.x(), pos.y(), pos.z(), 1 / lm_first.depth);
      }
    } else {
      memcpy(landmark_state[lm_id], lm.position.data(),
             sizeof(state_type) * POS_SIZE);
    }
    lm.flag = LandmarkFlag::INITIALIZED;
  } else if (lm.track.size() >= params->landmark_estimate_tracks ||
             lm.isMultiCamera()) {
    // Initialize by motion.
    std::vector<Swarm::Pose> poses;
    std::vector<Vector3d> points;
    auto ext_base = state->getExtrinsic(lm_first.camera_id);
    Eigen::Vector3d _min = (firstFrame.odom.pose() * ext_base).pos();
    Eigen::Vector3d _max = (firstFrame.odom.pose() * ext_base).pos();
    for (auto &it : lm.track) {
      auto frame = *state->getFramebyId(it.frame_id);
      auto ext = state->getExtrinsic(it.camera_id);
      auto cam_pose = frame.odom.pose() * ext;
      poses.push_back(cam_pose);
      points.push_back(it.pt3d_norm);
      _min = _min.cwiseMin((frame.odom.pose() * ext).pos());
      _max = _max.cwiseMax((frame.odom.pose() * ext).pos());
    }
    if ((_max - _min).norm() > params->depth_estimate_baseline) {
      // Initialize by triangulation
      Vector3d point_3d(0., 0., 0.);
      double tri_err = triangulatePoint3DPts(poses, points, point_3d);
      if (tri_err < params->tri_max_err) {
        lm.position = point_3d;
        if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
          auto ptcam = (firstFrame.odom.pose() * ext_base).inverse() * point_3d;
          auto inv_dep = 1 / ptcam.norm();
          if (inv_dep < params->max_inv_dep) {
            lm.flag = LandmarkFlag::INITIALIZED;
            *landmark_state[lm_id] = inv_dep;
            if (params->debug_print_states) {
              SPDLOG_INFO(
                  "Landmark {} "
                  "tracks {} baseline {:.2f} by tri. P {:.3f} "
                  "{:.3f} {:.3f} inv_dep {:.3f} err {:.3f}\n",
                  lm_id, lm.track.size(), (_max - _min).norm(), point_3d.x(),
                  point_3d.y(), point_3d.z(), inv_dep, tri_err);
            }
          } else {
            lm.flag = LandmarkFlag::INITIALIZED;
            *landmark_state[lm_id] = params->default_inv_dep;
            if (params->debug_print_states) {
              SPDLOG_WARN(
                  "Initialize failed too far away: landmark "
                  "{} tracks {} baseline {:.2f} by "
                  "triangulation position {:.3f} {:.3f} {:.3f} "
                  "inv_dep {:.3f}",
                  lm_id, lm.track.size(), (_max - _min).norm(), point_3d.x(),
                  point_3d.y(), point_3d.z(), inv_dep);
            }
          }
          if (params->debug_print_states) {
            for (auto &it : lm.track) {
              auto frame = *state->getFramebyId(it.frame_id);
              auto ext = state->getExtrinsic(it.camera_id);
              auto cam_pose = frame.odom.pose() * ext;
              auto reproject_pos = cam_pose.inverse() * point_3d;
              reproject_pos.normalize();
              SPDLOG_INFO(
                  "Frame {} camera_id {} index {} cam pose: {}"
                  "pt3d norm {:.3f} {:.3f} {:.3f} reproject "
                  "{:.3f} {:.3f} {:.3f}",
                  it.frame_id, it.camera_id, it.camera_index,
                  cam_pose.toStr().c_str(), it.pt3d_norm.x(), it.pt3d_norm.y(),
                  it.pt3d_norm.z(), reproject_pos.x(), reproject_pos.y(),
                  reproject_pos.z());
            }
          }
        } else {
          lm.flag = LandmarkFlag::INITIALIZED;
          memcpy(landmark_state[lm_id], lm.position.data(),
                 sizeof(state_type) * POS_SIZE);
        }
        // Some debug code
      } else {
        if (params->debug_print_states) {
          SPDLOG_WARN(
              "Initialize "
              "failed too large triangle error: landmark {} "
              "tracks {} baseline {:.2f} by triangulation position "
              "{:.3f} {:.3f} {:.3f}",
              lm_id, lm.track.size(), (_max - _min).norm(), point_3d.x(),
              point_3d.y(), point_3d.z());
        }
      }
    } else {
      if (params->debug_print_states) {
        SPDLOG_WARN(
            "Initialize "
            "failed too short baseline: landmark {} tracks {} "
            "baseline {:.2f}",
            lm_id, lm.track.size(), (_max - _min).norm());
      }
    }
  }
}

void D2LandmarkManager::initialLandmarks(const D2EstimatorState *state) {
  const Guard lock(state_lock);
  int inited_count = 0;
  for (auto &it : landmark_db) {
    auto &lm = it.second;
    auto lm_id = it.first;
    // Set to unsolved
    lm.solver_flag = LandmarkSolverFlag::UNSOLVED;
    if (lm.flag < LandmarkFlag::ESTIMATED) {
      if (lm.track.size() == 0) {
        SPDLOG_ERROR(
            "Initialize landmark "
            "{} failed, no track.",
            lm_id);
        continue;
      }
      initialLandmarkState(lm, state);
      inited_count += 1;
    } else if (lm.flag == LandmarkFlag::ESTIMATED) {
      // Extracting depth from estimated pos
      inited_count += 1;
      if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
        auto lm_per_frame = landmark_db.at(lm_id).track[0];
        auto firstFrame = state->getFramebyId(lm_per_frame.frame_id);
        auto ext = state->getExtrinsic(lm_per_frame.camera_id);
        Vector3d pos_cam =
            (firstFrame->odom.pose() * ext).inverse() * lm.position;
        *landmark_state[lm_id] = 1.0 / pos_cam.norm();
      } else {
        memcpy(landmark_state[lm_id], lm.position.data(),
               sizeof(state_type) * POS_SIZE);
      }
    }
  }

  SPDLOG_DEBUG("Total {} initialized {}", landmark_db.size(), inited_count);
}

int D2LandmarkManager::outlierRejectionByScale(const D2EstimatorState *state,
                          const std::set<LandmarkIdType> &used_landmarks)
{
  int remove_count = 0;
  if (estimated_landmark_size < params->perform_outlier_rejection_num ||
      params->remove_scale_outlier_threshold <= 1.0) {
    return 0;
  }
  // Filter by scale
  std::vector<double> scales;
  std::map<LandmarkIdType, double> scales_map;
  for (const auto &[lm_id, lm] : landmark_db) {
    if (lm.flag == LandmarkFlag::ESTIMATED &&
      used_landmarks.find(lm_id) != used_landmarks.end()) {
      // Calculate the scale to track[0]
      auto lm_per_frame = lm.track[0];
      auto firstFrame = state->getFramebyId(lm_per_frame.frame_id);
      auto ext = state->getExtrinsic(lm_per_frame.camera_id);
      Vector3d pos_cam = (firstFrame->odom.pose() * ext).inverse() * lm.position;
      double scale = pos_cam.norm();
      scales.push_back(scale);
      scales_map[lm_id] = scale;
    }
  }

  // Filter by nxmiddle point
  double filter_out_thres = 0;
  if (scales.size() > 0) {
    std::sort(scales.begin(), scales.end());
    double scale_median = scales[scales.size() / 2];
    filter_out_thres = scale_median * params->remove_scale_outlier_threshold;
    SPDLOG_DEBUG("Scale median {:.2f} thres {:.2f}", scale_median,
                filter_out_thres);
    for (const auto& [lm_id, scale] : scales_map) {
      if (scale > filter_out_thres) {
        auto &lm = landmark_db.at(lm_id);
        lm.flag = LandmarkFlag::OUTLIER;
        remove_count++;
        SPDLOG_DEBUG("remove LM {} scale {:.2f} thres {:.2f}", lm_id, scale,
                    filter_out_thres);
      }
    }
  }
  SPDLOG_INFO("outlierRejectionByScale remove {}/{} landmarks", remove_count,
              scales.size());
  return remove_count;
}

int D2LandmarkManager::outlierRejection(
    const D2EstimatorState *state,
    const std::set<LandmarkIdType> &used_landmarks) {
  const Guard lock(state_lock);
  int total_count = 0;
  if (estimated_landmark_size < params->perform_outlier_rejection_num) {
    return 0;
  }

  int remove_count = outlierRejectionByScale(state, used_landmarks);
    
  for (auto &it : landmark_db) {
    auto &lm = it.second;
    auto lm_id = it.first;
    if (lm.flag == LandmarkFlag::ESTIMATED &&
        used_landmarks.find(lm_id) != used_landmarks.end()) {
      double err_sum = 0;
      double err_cnt = 0;
      int count_err_track = 0;
      total_count++;
      for (auto it = lm.track.begin() + 1; it != lm.track.end();) {
        auto pose = state->getFramebyId(it->frame_id)->odom.pose();
        auto ext = state->getExtrinsic(it->camera_id);
        auto pt3d_n = it->pt3d_norm;
        Vector3d pos_cam = (pose * ext).inverse() * lm.position;
        pos_cam.normalize();
        // Compute reprojection error
        Vector3d reproj_error = pt3d_n - pos_cam;
        if (reproj_error.norm() * params->focal_length >
            params->landmark_outlier_threshold) {
          count_err_track += 1;
          ++it;
        } else {
          ++it;
        }
        err_sum += reproj_error.norm();
        err_cnt += 1;
      }
      lm.num_outlier_tracks = count_err_track;
      if (err_cnt > 0) {
        double reproj_err = err_sum / err_cnt;
        if (reproj_err * params->focal_length >
            params->landmark_outlier_threshold) {
          remove_count++;
          lm.flag = LandmarkFlag::OUTLIER;
          SPDLOG_DEBUG(
              "remove LM {} inv_dep/dep "
              "{:.2f}/{:.2f} pos {:.2f} {:.2f} {:.2f} reproj_error "
              "{:.2f}",
              lm_id, *landmark_state[lm_id], 1. / (*landmark_state[lm_id]),
              lm.position.x(), lm.position.y(), lm.position.z(),
              reproj_err * params->focal_length);
        }
      }
    }
  }
  SPDLOG_DEBUG("outlierRejection remove {}/{} landmarks", remove_count,
                total_count);
  return remove_count;
}

void D2LandmarkManager::syncState(const D2EstimatorState *state) {
  const Guard lock(state_lock);
  // Sync inverse depth to 3D positions
  estimated_landmark_size = 0;
  for (auto it : landmark_state) {
    auto lm_id = it.first;
    auto &lm = landmark_db.at(lm_id);
    if (lm.solver_flag == LandmarkSolverFlag::SOLVED) {
      if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
        auto inv_dep = *it.second;
        if (inv_dep < 0) {
          SPDLOG_DEBUG("[Warn] small inv dep {:.2f} found", inv_dep);
          lm.flag = LandmarkFlag::OUTLIER;
          continue;
        }
        if (inv_dep > params->max_inv_dep) {
          inv_dep = params->default_inv_dep;
        }
        auto lm_per_frame = lm.track[0];
        const auto &firstFrame = state->getFramebyId(lm_per_frame.frame_id);
        auto ext = state->getExtrinsic(lm_per_frame.camera_id);
        auto pt3d_n = lm_per_frame.pt3d_norm;
        Vector3d pos = pt3d_n / inv_dep;
        pos = firstFrame->odom.pose() * ext * pos;
        lm.position = pos;
        lm.flag = LandmarkFlag::ESTIMATED;
        // SPDLOG_DEBUG(
        //     "update LM {:d} inv_dep/dep "
        //     "{:.2f}/{:.2f} depmea {:d} {:.2f} pt3d_n {:.2f} {:.2f} "
        //     "{:.2f} pos "
        //     "{:.2f} {:.2f} {:.2f} baseFrame {:d} pose {} extrinsic {}",
        //     lm_id, inv_dep, 1. / inv_dep, lm_per_frame.depth_mea,
        //     lm_per_frame.depth, pt3d_n.x(), pt3d_n.y(), pt3d_n.z(),
        //     pos.x(), pos.y(), pos.z(), lm_per_frame.frame_id,
        //     firstFrame->odom.pose().toStr().c_str(),
        //     ext.toStr().c_str());
      } else {
        lm.position.x() = it.second[0];
        lm.position.y() = it.second[1];
        lm.position.z() = it.second[2];
        lm.flag = LandmarkFlag::ESTIMATED;
      }
      estimated_landmark_size++;
    }
  }
}

void D2LandmarkManager::removeLandmark(const LandmarkIdType &id) {
  landmark_db.erase(id);
  landmark_state.erase(id);
}

double triangulatePoint3DPts(const std::vector<Swarm::Pose> poses,
                             const std::vector<Vector3d> &points,
                             Vector3d &point_3d) {
  MatrixXd design_matrix(poses.size() * 2, 4);
  assert(poses.size() > 0 && poses.size() == points.size() &&
         "We at least have 2 poses and number of pts and poses must equal");
  for (unsigned int i = 0; i < poses.size(); i++) {
    double norm = points[i].norm();
    double p0x = points[i][0] / norm;
    double p0y = points[i][1] / norm;
    double p0z = points[i][2] / norm;
    Eigen::Matrix<double, 3, 4> pose;
    auto R0 = poses[i].R();
    auto t0 = poses[i].pos();
    pose.leftCols<3>() = R0.transpose();
    pose.rightCols<1>() = -R0.transpose() * t0;
    design_matrix.row(i * 2) = p0x * pose.row(2) - p0z * pose.row(0);
    design_matrix.row(i * 2 + 1) = p0y * pose.row(2) - p0z * pose.row(1);
  }
  Vector4d triangulated_point;
  triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);

  double sum_err = 0;
  double err_pose_0 = 0.0;
  for (unsigned int i = 0; i < poses.size(); i++) {
    auto reproject_pos = poses[i].inverse() * point_3d;
    reproject_pos.normalize();
    Vector3d err = points[i].normalized() - reproject_pos;
    if (i == 0) {
      err_pose_0 = err.norm();
    }
    sum_err += err.norm();
  }
  return sum_err / points.size() + err_pose_0;
}

std::map<FrameIdType, Swarm::Pose> D2LandmarkManager::SFMInitialization(
    const std::vector<VINSFrame *> frames, int camera_idx) {
  SPDLOG_DEBUG("SFMInitialization with camera {}", camera_idx);

  // TODO: Add camera param? or consider multi-camera case
  // Here we assume we are using mono camera
  // First we init with solve 5 points use last two frames
  std::map<FrameIdType, Swarm::Pose> initial;
  assert(frames.size() > 2);
  auto last_frame = frames[frames.size() - 1];
  VINSFrame *head_frame_for_match = nullptr;
  Swarm::Pose relative_pose;
  // Start the scale with solve5pts
  for (size_t i = 0; i < frames.size() - 1; i++) {
    head_frame_for_match = frames[i];
    if (SolveRelativePose5Pts(relative_pose, camera_idx, last_frame->frame_id,
                              head_frame_for_match->frame_id)) {
      initial[last_frame->frame_id] = Swarm::Pose::Identity();
      initial[head_frame_for_match->frame_id] = relative_pose;
      SPDLOG_INFO("Frame_id {} PnP result: {}", last_frame->frame_id,
                  initial[last_frame->frame_id].toStr());
      SPDLOG_INFO("Frame_id {} PnP result: {}", head_frame_for_match->frame_id,
                  relative_pose.toStr());
      break;
    } else {
      continue;
    }
  }
  if (initial.size() == 0) {
    SPDLOG_WARN("SFMInitialization failed");
    return initial;
  }

  // First triangulation
  auto last_triangluation_pts =
      triangulationFrames(last_frame->frame_id, initial[last_frame->frame_id],
                          head_frame_for_match->frame_id,
                          initial[head_frame_for_match->frame_id], camera_idx);

  // Recursive triangluation
  for (int i = frames.size() - 2; i >= 0; i--) {
    auto frame = frames[i];
    if (frame->frame_id == head_frame_for_match->frame_id) {
      continue;
    }
    Swarm::Pose pose = Swarm::Pose::Identity();
    // Found the landmarks observerd by frame and in points3d
    if (InitFramePoseWithPts(pose, last_triangluation_pts, frame->frame_id,
                             camera_idx)) {
      // Triangulate all the common points
      initial[frame->frame_id] = pose;
      last_triangluation_pts = triangulationFrames(initial, camera_idx, 2);
      SPDLOG_INFO("{} points initialized", last_triangluation_pts.size());
    } else {
      return std::map<FrameIdType, Swarm::Pose>();
    }
  }

  // Now re-triangluation all points
  auto initial_pts = triangulationFrames(initial, camera_idx, 3);
  auto ret = PerformBA(initial, last_frame, head_frame_for_match, initial_pts,
                       camera_idx);
  return ret;
}

const std::map<FrameIdType, Swarm::Pose> D2LandmarkManager::PerformBA(
    const std::map<FrameIdType, Swarm::Pose> &initial, VINSFrame *last_frame,
    VINSFrame *head_frame_for_match,
    std::map<LandmarkIdType, Vector3d> initial_pts, int camera_idx) const {
  SPDLOG_INFO("{} points initialized. Now start BA", initial_pts.size());
  std::map<FrameIdType, Swarm::Pose> ret;

  std::map<FrameIdType, double *> c_translation, c_rotation;
  std::map<LandmarkIdType, double *> points;

  ceres::Problem problem;
  ceres::LocalParameterization *local_parameterization =
      new ceres::EigenQuaternionParameterization();
  ceres::HuberLoss *loss = new ceres::HuberLoss(1.0);

  for (auto it : initial) {
    // Initial states
    auto frame_id = it.first;
    c_translation[frame_id] = new double[3];
    c_rotation[frame_id] = new double[4];
    Eigen::Map<Eigen::Vector3d> pos(c_translation[frame_id]);
    Eigen::Map<Eigen::Quaterniond> q(c_rotation[frame_id]);
    pos = it.second.pos();
    q = it.second.att();
    problem.AddParameterBlock(c_rotation[frame_id], 4, local_parameterization);
    problem.AddParameterBlock(c_translation[frame_id], 3);
    if (frame_id == last_frame->frame_id) {
      problem.SetParameterBlockConstant(c_rotation[frame_id]);
    }
    if (frame_id == last_frame->frame_id ||
        frame_id == head_frame_for_match->frame_id) {
      // For scale
      problem.SetParameterBlockConstant(c_translation[frame_id]);
    }
  }

  for (auto &it : landmark_db) {
    auto &lm = it.second;
    auto lm_id = it.first;
    std::vector<Swarm::Pose> poses;
    std::vector<Eigen::Vector3d> pt3d_norms;
    if (initial_pts.count(lm_id) == 0 ||
        lm.track.size() < params->landmark_estimate_tracks) {
      continue;
    }
    points[lm_id] = new double[3];
    Eigen::Map<Eigen::Vector3d> pt3d(points[lm_id]);
    pt3d = initial_pts.at(lm_id);

    for (auto &it : lm.track) {
      if (it.camera_id == camera_idx && initial.count(it.frame_id) > 0) {
        const auto &pt3d_norm = it.pt3d_norm;
        ceres::CostFunction *cost_function = ReprojectionError3D::Create(
            pt3d_norm.x() / pt3d_norm.z(), pt3d_norm.y() / pt3d_norm.z());
        problem.AddResidualBlock(cost_function, loss, c_rotation[it.frame_id],
                                 c_translation[it.frame_id], points.at(lm_id));
      }
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.max_solver_time_in_seconds = 0.2;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  SPDLOG_INFO("Finish solve BA in {:.2f}ms. rpt {}",
              summary.total_time_in_seconds * 1000.0, summary.BriefReport());
  Swarm::Pose pose0_inv = Swarm::Pose::Identity();
  bool is_pose0_set = false;
  for (auto it : initial) {
    // Initial states
    auto frame_id = it.first;
    Eigen::Map<Eigen::Vector3d> pos(c_translation[frame_id]);
    Eigen::Map<Eigen::Quaterniond> quat(c_rotation[frame_id]);
    Swarm::Pose camera_pose(pos, quat);
    if (!is_pose0_set) {
      pose0_inv = camera_pose.inverse();
      is_pose0_set = true;
    }
    ret[frame_id] = pose0_inv * camera_pose;
    SPDLOG_INFO("SfM init {}: Cam {}", frame_id, ret[frame_id].toStr());
  }
  return ret;
}

bool D2LandmarkManager::InitFramePoseWithPts(
    Swarm::Pose &ret,
    std::map<LandmarkIdType, Vector3d> &last_triangluation_pts,
    FrameIdType frame_id, int camera_idx) {
  auto landmark_ids = getRelatedLandmarks(frame_id);
  std::vector<cv::Point3f> points_3d;
  std::vector<cv::Point2f> points_undist;
  std::vector<LandmarkIdType> landmark_ids_used;
  for (auto lm_id : landmark_ids) {
    if (last_triangluation_pts.find(lm_id) != last_triangluation_pts.end()) {
      const auto &pt3d = last_triangluation_pts.at(lm_id);
      points_3d.emplace_back(pt3d.x(), pt3d.y(), pt3d.z());
      auto lm = landmark_db.at(lm_id);
      auto lm_per_frame = lm.at(frame_id);
      cv::Point2f pt_undist(
          lm_per_frame.pt3d_norm.x() / lm_per_frame.pt3d_norm.z(),
          lm_per_frame.pt3d_norm.y() / lm_per_frame.pt3d_norm.z());
      points_undist.push_back(pt_undist);
      landmark_ids_used.push_back(lm_id);
    }
  }

  // Then use cv::solvePnPRansac to solve the pose of frame
  if (points_undist.size() < 5) {
    SPDLOG_ERROR(
        "PnP failed in "
        "SFMInitialization on {}, "
        "only {} pts",
        frame_id, points_3d.size());
    return false;
  }
  cv::Mat rvec, tvec;
  D2FrontEnd::PnPInitialFromCamPose(ret, rvec, tvec);
  cv::Mat K = cv::Mat::eye(3, 3, CV_64F);  // Use undist point, so identity
  std::vector<uint8_t> inliers;
  cv::solvePnPRansac(points_3d, points_undist, K, cv::Mat(), rvec, tvec, true,
                     100, 0.01, 0.99, inliers);
  // Convert to eigen
  ret = D2FrontEnd::PnPRestoCamPose(rvec, tvec);
  // Print result
  int num_inliers = 0;
  for (size_t i = 0; i < inliers.size(); i++) {
    if (inliers[i]) {
      num_inliers++;
    } else {
      last_triangluation_pts.erase(landmark_ids_used[i]);
    }
  }
  SPDLOG_INFO("Frame_id {} PnP result: {} inlier {}/{}", frame_id, ret.toStr(),
              num_inliers, points_undist.size());
  return true;
}

bool D2LandmarkManager::SolveRelativePose5Pts(Swarm::Pose &ret, int camera_idx,
                                              FrameIdType frame1_id,
                                              FrameIdType frame2_id) {
  // Get their landmarks and find the common
  auto common_lm = findCommonLandmarkPerFrames(frame1_id, frame2_id);

  // Solve with 5 pts method
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
  double sum_parallex = 0.0;
  for (auto &it : common_lm) {
    auto &lm1 = it.first;
    auto &lm2 = it.second;
    // Check if the camera index is the same, if not we will skip this
    // landmark
    if (lm1.camera_id != camera_idx || lm2.camera_id != camera_idx) {
      continue;
    }
    // Use pt3d_norm
    corres.emplace_back(lm1.pt3d_norm, lm2.pt3d_norm);
    sum_parallex += (lm1.pt3d_norm - lm2.pt3d_norm).norm();
    // Draw the arrow of the points using pt2d, it's already cv::Point2f
  }
  if (corres.size() < params->solve_relative_pose_min_pts ||
      sum_parallex / corres.size() < params->solve_relative_pose_min_parallex) {
    SPDLOG_WARN(
        "Solve 5 pts failed, only {} "
        "pts, parallex {:.2f}",
        corres.size(), sum_parallex / corres.size());
    return false;
  }
  utils::MotionEstimator estimator;
  Matrix3d R;
  Vector3d T;
  if (!estimator.solveRelativeRT(corres, R, T)) {
    SPDLOG_WARN("Solve 5 pts failed");
    return false;
  }
  ret = Swarm::Pose(R, T);
  SPDLOG_INFO(
      "Frame {}-{} Solve 5 pts with {} pts result: "
      "{}",
      frame1_id, frame2_id, corres.size(), ret.toStr());
  return true;
}

std::map<FrameIdType, Vector3d> D2LandmarkManager::triangulationFrames(
    FrameIdType frame1_id, const Swarm::Pose &pose1, FrameIdType frame2_id,
    const Swarm::Pose &pose2, int camera_idx) {
  auto common_lm = findCommonLandmarkPerFrames(frame1_id, frame2_id);
  // triangluate these points use the pose
  std::map<FrameIdType, Vector3d> points3d;
  for (auto &it : common_lm) {
    auto &lm1 = it.first;
    auto &lm2 = it.second;
    if (lm1.camera_id != camera_idx || lm2.camera_id != camera_idx) {
      continue;
    }
    std::vector<Vector3d> points{lm1.pt3d_norm, lm2.pt3d_norm};
    std::vector<Swarm::Pose> poses{pose1, pose2};
    // Perform triangulation
    Vector3d point_3d(0., 0., 0.);
    auto ret = triangulatePoint3DPts(poses, points, point_3d);
    if (ret < params->mono_initial_tri_max_err)
      points3d[lm1.landmark_id] = point_3d;
  }
  return points3d;
}

std::map<LandmarkIdType, Vector3d> D2LandmarkManager::triangulationFrames(
    const std::map<FrameIdType, Swarm::Pose> &frame_poses, int camera_idx,
    int min_tracks) {
  std::map<LandmarkIdType, Vector3d> ret;
  for (auto &it : landmark_db) {
    auto &lm = it.second;
    auto lm_id = it.first;
    std::vector<Swarm::Pose> poses;
    std::vector<Eigen::Vector3d> pt3d_norms;
    if (lm.track.size() < min_tracks) {
      continue;
    }
    for (auto &it : lm.track) {
      if (it.camera_id == camera_idx && frame_poses.count(it.frame_id) > 0) {
        poses.emplace_back(frame_poses.at(it.frame_id));
        pt3d_norms.emplace_back(it.pt3d_norm);
      }
    }
    if (poses.size() < min_tracks) {
      continue;
    }
    // Perform triangulation
    Vector3d point_3d(0., 0., 0.);
    auto err = triangulatePoint3DPts(poses, pt3d_norms, point_3d);
    if (point_3d.norm() > 1e-2 && err < params->mono_initial_tri_max_err) {
      ret[lm.landmark_id] = point_3d;
    }
  }
  return ret;
}

}  // namespace D2VINS
