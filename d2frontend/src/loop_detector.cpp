#include <d2common/d2vinsframe.h>
#include <d2frontend/CNN/superglue_onnx.h>
#include <d2frontend/loop_cam.h>
#include <d2frontend/loop_detector.h>
#include <d2frontend/utils.h>
#include <d2frontend/feature_matcher.h>
#include <d2frontend/pnp_utils.h>
#include <faiss/IndexFlat.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <swarm_msgs/relative_measurments.hpp>

using namespace std::chrono;
using namespace D2Common;

#define USE_FUNDMENTAL
#define MAX_LOOP_ID 100000000

namespace D2FrontEnd {

void LoopDetector::processImageArray(VisualImageDescArray &image_array) {
  // Lock frame_mutex with Guard
  std::lock_guard<std::recursive_mutex> guard(frame_mutex);
  TicToc tt;
  static double t_sum = 0;
  static int t_count = 0;

  auto start = high_resolution_clock::now();

  if (t0 < 0) {
    t0 = image_array.stamp;
  }

  SPDLOG_INFO(
      "[LoopDetector] processImageArray {} from {} images: {} landmark: {} "
      "lazy: {} matched_to {}@D{}",
      image_array.frame_id, image_array.drone_id, image_array.images.size(),
      image_array.spLandmarkNum(), image_array.is_lazy_frame,
      image_array.matched_frame, image_array.matched_drone);

  if (image_array.images.size() == 0) {
    SPDLOG_WARN("[LoopDetector] FlattenDesc must carry more than zero images");
    return;
  }

  ego_motion_traj.push(ros::Time(image_array.stamp), image_array.pose_drone);

  int drone_id = image_array.drone_id;
  int images_num = image_array.images.size();
  bool is_matched_frame = image_array.isMatchedFrame();
  bool is_lazy_frame = image_array.is_lazy_frame;
  if (is_matched_frame && image_array.matched_drone != self_id) {
    if (!image_array.is_lazy_frame) {
      // Will be cache to databse
      addImageArrayToDatabase(image_array, false);
      printf(
          "[LoopDetector@%d] Add KF %ld from drone %d images: %d landmark: %d "
          "lazy: %d matched_to %d",
          self_id, image_array.frame_id, drone_id, image_array.images.size(),
          image_array.spLandmarkNum(), image_array.is_lazy_frame,
          image_array.matched_frame);
    }
    // printf("[LoopDetector@%d] Frame %ld matched to drone %ld, giveup\n",
    // self_id, image_array.frame_id, image_array.matched_drone);
    return;
  }

  if (drone_id != this->self_id && databaseSize() == 0) {
    ROS_INFO("[LoopDetector] Empty local database, will giveup remote image");
    return;
  }

  bool new_node = all_nodes.find(image_array.drone_id) == all_nodes.end();
  all_nodes.insert(image_array.drone_id);

  int dir_count = 0;
  for (auto &img : image_array.images) {
    if (img.spLandmarkNum() > 0 || image_array.is_lazy_frame) {
      dir_count++;
    }
  }
  if (dir_count < _config.MIN_DIRECTION_LOOP) {
    ROS_INFO(
        "[LoopDetector@%d] Give up image_array %ld with less than %d(%d) "
        "available images",
        self_id, image_array.frame_id, _config.MIN_DIRECTION_LOOP, dir_count);
    return;
  }

  if (image_array.spLandmarkNum() >= _config.loop_inlier_feature_num ||
      is_lazy_frame) {
    // Initialize images for visualization
    if (params->show) {
      std::vector<cv::Mat> imgs;
      for (unsigned int i = 0; i < images_num; i++) {
        auto &img_des = image_array.images[i];
        if (!img_des.raw_image.empty()) {
          imgs.emplace_back(img_des.raw_image);
        } else if (img_des.image.size() != 0) {
          imgs.emplace_back(decode_image(img_des));
        } else {
          // imgs[i] = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255,
          // 255));
          imgs.emplace_back(
              cv::Mat(params->height, params->width, CV_8U, cv::Scalar(255)));
        }
        if (params->camera_configuration == STEREO_PINHOLE) {
          break;
        }
      }
      msgid2cvimgs[image_array.frame_id] = imgs;
    }

    bool success = false;
    VisualImageDescArray _old_fisheye_img;
    int camera_index = 1;
    int camera_index_old = -1;
    if (is_matched_frame) {
      if (!hasFrame(image_array.matched_frame)) {
        success = false;
        printf(
            "[LoopDetector] frame %ld is matched to local frame %ld but not in "
            "db\n",
            image_array.frame_id, image_array.matched_frame);
      } else {
        // printf("[LoopDetector] frame %ld is matched to local frame %ld in
        // db\n", image_array.frame_id, image_array.matched_frame);
        success = true;
        const std::lock_guard<std::mutex> lock(keyframe_database_mutex);
        _old_fisheye_img = keyframe_database.at(image_array.matched_frame);
        camera_index = 0;  // TODO: this is a hack
        camera_index_old = 0;
        if (is_lazy_frame) {
          // In this case, it's a keyframe that has been broadcasted and should
          // be recorded in database
          printf(
              "[LoopDetector] frame %ld is matched to local frame %ld in db "
              "and we find it in cache\n",
              image_array.frame_id, image_array.matched_frame);
          if (keyframe_database.find(image_array.frame_id) ==
              keyframe_database.end()) {
            ROS_WARN(
                "[LoopDetector] Lazy frame %ld is matched to local frame %ld "
                "in db, but is not in cache",
                image_array.frame_id, image_array.matched_frame);
            success = false;
          } else {
            printf("[LoopDetector] frame %ld is found in database\n",
                   image_array.frame_id);
            image_array = keyframe_database.at(image_array.frame_id);
          }
        }
      }
    } else {
      if (databaseSize() > _config.match_index_dist ||
          drone_id != self_id &&
              databaseSize() > _config.match_index_dist_remote) {
        success = queryImageArrayFromDatabase(image_array, _old_fisheye_img,
                                              camera_index, camera_index_old);
        auto stop = high_resolution_clock::now();
      }
    }
    if (success) {
      if (!is_matched_frame && is_lazy_frame) {
        // In this case, we need to send the matched frame to the drone
        _old_fisheye_img.matched_drone = image_array.drone_id;
        _old_fisheye_img.matched_frame = image_array.frame_id;
        printf(
            "[LoopDetector@%d] Lazy frame %d is matched with %d try to "
            "broadcast this frame\n",
            self_id, image_array.frame_id, _old_fisheye_img.frame_id);
        if (broadcast_keyframe_cb) {
          broadcast_keyframe_cb(_old_fisheye_img);
        }
      } else {
        printf("Compute loop connection %ld and %ld\n", image_array.frame_id,
               _old_fisheye_img.frame_id);
        swarm_msgs::LoopEdge ret;
        if (_old_fisheye_img.drone_id == self_id) {
          success = computeLoop(_old_fisheye_img, image_array, camera_index_old,
                                camera_index, ret);
        } else if (image_array.drone_id == self_id) {
          success = computeLoop(image_array, _old_fisheye_img, camera_index,
                                camera_index_old, ret);
        } else {
          ROS_WARN("[LoopDetector%d] Will not compute loop, drone id is %d",
                   self_id, image_array.drone_id);
        }
        if (success) {
          onLoopConnection(ret);
        }
      }
    } else {
      if (params->verbose)
        printf("[LoopDetector@%d] No matched image for frame %ld\n", self_id,
               image_array.frame_id);
    }
    if (!is_lazy_frame && (!image_array.prevent_adding_db || new_node)) {
      if (image_array.drone_id == self_id) {
        // Only add local frame to database
        addImageArrayToDatabase(image_array, true);
      } else {
        addImageArrayToDatabase(image_array, false);
      }
    }
  }

  t_sum += tt.toc();
  t_count += 1;
  if (params->verbose || params->enable_perf_output)
    printf("[LoopDetector] Full LoopDetect avg %.1fms cur %.1fms\n",
           t_sum / t_count, tt.toc());
}

cv::Mat LoopDetector::decode_image(const VisualImageDesc &_img_desc) {
  auto start = high_resolution_clock::now();
  // auto ret = cv::imdecode(_img_desc.image, cv::IMREAD_GRAYSCALE);
  auto ret = cv::imdecode(_img_desc.image, cv::IMREAD_UNCHANGED);
  // std::cout << "IMDECODE Cost " <<
  // duration_cast<microseconds>(high_resolution_clock::now() -
  // start).count()/1000.0 << "ms" << std::endl;

  return ret;
}

int LoopDetector::addImageArrayToDatabase(
    VisualImageDescArray &new_fisheye_desc, bool add_to_faiss) {
  if (add_to_faiss) {
    for (size_t i = 0; i < new_fisheye_desc.images.size(); i++) {
      auto &img_desc = new_fisheye_desc.images[i];
      if (img_desc.spLandmarkNum() > 0 && img_desc.image_desc.size() > 0) {
        int index = addImageDescToDatabase(img_desc);
        index_to_frame_id[index] = new_fisheye_desc.frame_id;
        imgid2dir[index] = i;
      }
      if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
        break;
      }
      const std::lock_guard<std::mutex> lock(keyframe_database_mutex);
    }
  }
  keyframe_database[new_fisheye_desc.frame_id] = new_fisheye_desc;
  SPDLOG_INFO(
      "[LoopDetector] Add KF {} with {} images from {} to local keyframe "
      "database. Total frames: {}",
      new_fisheye_desc.frame_id, new_fisheye_desc.images.size(),
      new_fisheye_desc.drone_id, keyframe_database.size());
  // new_fisheye_desc.printSize();
  return new_fisheye_desc.frame_id;
}

int LoopDetector::addImageDescToDatabase(VisualImageDesc &img_desc_a) {
  if (img_desc_a.drone_id == self_id) {
    local_index.add(1, img_desc_a.image_desc.data());
    return local_index.ntotal - 1;
  } else {
    remote_index.add(1, img_desc_a.image_desc.data());
    return remote_index.ntotal - 1 + REMOTE_MAGIN_NUMBER;
  }
  return -1;
}

int LoopDetector::queryFrameIndexFromDatabase(const VisualImageDesc &img_desc,
                                              double &similarity) {
  double thres = _config.loop_detection_netvlad_thres;
  int ret = -1;
  if (img_desc.drone_id == self_id) {
    // Then this is self drone
    double similarity_local, similarity_remote;
    int ret_remote =
        queryIndexFromDatabase(img_desc, remote_index, true, thres,
                               _config.match_index_dist, similarity_remote);
    int ret_local =
        queryIndexFromDatabase(img_desc, local_index, false, thres,
                               _config.match_index_dist, similarity_local);
    if (ret_remote >= 0 && ret_local >= 0) {
      if (similarity_local > similarity_remote) {
        similarity = similarity_local;
        return ret_local;
      } else {
        similarity = similarity_remote;
        return similarity_remote;
      }
    } else if (ret_remote >= 0) {
      similarity = similarity_remote;
      return similarity_remote;
    } else if (ret_local >= 0) {
      similarity = similarity_local;
      return ret_local;
    }
  } else {
    ret = queryIndexFromDatabase(img_desc, local_index, false, thres,
                                 _config.match_index_dist_remote, similarity);
  }
  return ret;
}

int LoopDetector::queryIndexFromDatabase(const VisualImageDesc &img_desc,
                                         faiss::IndexFlatIP &index,
                                         bool remote_db, double thres,
                                         int max_index, double &similarity) {
  float similiarity[1024] = {0};
  faiss::idx_t labels[1024];

  int index_offset = 0;
  if (remote_db) {
    index_offset = REMOTE_MAGIN_NUMBER;
  }
  for (int i = 0; i < 1000; i++) {
    labels[i] = -1;
  }
  int search_num = std::min(SEARCH_NEAREST_NUM + max_index, (int)index.ntotal);
  if (search_num <= 0) {
    return -1;
  }
  index.search(1, img_desc.image_desc.data(), search_num, similiarity, labels);
  int return_frame_id = -1, return_drone_id = -1;
  int k = -1;
  for (int i = 0; i < search_num; i++) {
    if (labels[i] < 0) {
      continue;
    }
    if (index_to_frame_id.find(labels[i] + index_offset) ==
        index_to_frame_id.end()) {
      ROS_WARN("[LoopDetector] Can't find image %d; skipping",
               labels[i] + index_offset);
      continue;
    }
    // int return_frame_id = index_to_frame_id.at(labels[i] + index_offset);
    return_frame_id = labels[i] + index_offset;
    const std::lock_guard<std::mutex> lock(keyframe_database_mutex);
    return_drone_id =
        keyframe_database.at(index_to_frame_id.at(return_frame_id)).drone_id;
    // ROS_INFO("Return Label %d/%d/%d from %d, distance %f/%f", labels[i] +
    // index_offset, index.ntotal, index.ntotal - max_index , return_drone_id,
    // similiarity[i], thres);
    if (labels[i] <= index.ntotal - max_index && similiarity[i] > thres) {
      // Is same id, max index make sense
      k = i;
      thres = similarity = similiarity[i];
      return return_frame_id;
    }
  }
  // ROS_INFO("Database return %ld on drone %d, radius %f frame_id %d",
  // labels[k] + index_offset, return_drone_id, similiarity[k],
  // return_frame_id);
  return -1;
}

bool LoopDetector::queryImageArrayFromDatabase(
    const VisualImageDescArray &img_desc_a, VisualImageDescArray &ret,
    int &camera_index_new, int &camera_index_old) {
  double best_similarity = -1;
  int best_image_index = -1;
  // Strict use camera_index 1 now
  camera_index_new = 0;
  if (loop_cam->getCameraConfiguration() == CameraConfig::STEREO_FISHEYE) {
    camera_index_new = 1;
  } else if (loop_cam->getCameraConfiguration() ==
             CameraConfig::FOURCORNER_FISHEYE) {
    // If four coner fishe, use camera 2.
    camera_index_new = 2;
  } else if (loop_cam->getCameraConfiguration() ==
                 CameraConfig::STEREO_PINHOLE ||
             loop_cam->getCameraConfiguration() ==
                 CameraConfig::PINHOLE_DEPTH || 
                 loop_cam->getCameraConfiguration() == CameraConfig::MONOCULAR) {
    camera_index_new = 0;
  } else {
    SPDLOG_ERROR(
        "[LoopDetector] Camera configuration {} not support yet in "
        "queryImageArrayFromDatabase",
        loop_cam->getCameraConfiguration());
  }

  if (img_desc_a.images[camera_index_new].spLandmarkNum() > 0 && img_desc_a.images[camera_index_new].hasImageDesc() ||
      img_desc_a.is_lazy_frame) {
    double similarity = -1;
    int index = queryFrameIndexFromDatabase(
        img_desc_a.images.at(camera_index_new), similarity);
    if (index != -1 && similarity > best_similarity) {
      best_image_index = index;
      best_similarity = similarity;
    }

    if (best_image_index != -1) {
      const std::lock_guard<std::mutex> lock(keyframe_database_mutex);
      int frame_id = index_to_frame_id[best_image_index];
      camera_index_old = imgid2dir[best_image_index];
      printf(
          "[LoopDetector] Query image for %ld: ret frame_id %d index %d drone "
          "%d with camera %d similarity %f\n",
          img_desc_a.frame_id, frame_id, best_image_index,
          keyframe_database.at(frame_id).drone_id, camera_index_old,
          best_similarity);
      ret = keyframe_database.at(frame_id);
      return true;
    }
  }

  camera_index_old = -1;
  ret.frame_id = -1;
  return false;
}

int LoopDetector::databaseSize() const {
  return local_index.ntotal + remote_index.ntotal;
}

bool LoopDetector::checkLoopOdometryConsistency(LoopEdge &loop_conn) const {
  if (loop_conn.drone_id_a != loop_conn.drone_id_b || _config.DEBUG_NO_REJECT) {
    // Is inter_loop, odometry consistency check is disabled.
    return true;
  }

  Swarm::LoopEdge edge(loop_conn);
  auto odom =
      ego_motion_traj.get_relative_pose_by_appro_ts(edge.ts_a, edge.ts_b);
  Eigen::Matrix6d cov_vec = odom.second + edge.getCovariance();
  auto dp = Swarm::Pose::DeltaPose(edge.relative_pose, odom.first);
  auto md = Swarm::computeSquaredMahalanobisDistance(dp.log_map(), cov_vec);
  if (md > _config.odometry_consistency_threshold) {
    printf(
        "[LoopDetector] LoopEdge-Odometry consistency check failed %.1f, odom "
        "%s loop %s dp %s.\n",
        md, odom.first.toStr().c_str(), edge.relative_pose.toStr().c_str(),
        dp.toStr().c_str());
    return false;
  }

  printf(
      "[LoopDetector] LoopEdge-Odometry consistency OK %.1f odom %s loop %s dp "
      "%s.\n",
      md, odom.first.toStr().c_str(), edge.relative_pose.toStr().c_str(),
      dp.toStr().c_str());
  return true;
}

// Note! here the norms are both projected to main dir's unit sphere.
// index2dirindex store the dir and the index of the point
bool LoopDetector::computeCorrespondFeaturesOnImageArray(
    const VisualImageDescArray &frame_array_a,
    const VisualImageDescArray &frame_array_b, int main_dir_a, int main_dir_b,
    std::vector<Vector3d> &lm_pos_a, std::vector<Vector3d> &lm_norm_3d_b,
    std::vector<int> &cam_indices,
    std::vector<std::pair<int, int>> &index2dirindex_a,
    std::vector<std::pair<int, int>> &index2dirindex_b) {
  std::vector<int> dirs_a;
  std::vector<int> dirs_b;

  if (params->camera_configuration == STEREO_PINHOLE) {
    dirs_a = {0};
    dirs_b = {0};
  } else if (params->camera_configuration == PINHOLE_DEPTH) {
    dirs_a = {0, 1};
    dirs_b = {0};
  } else if (params->camera_configuration == STEREO_FISHEYE ||
             params->camera_configuration == FOURCORNER_FISHEYE) {
    for (int _dir_a = main_dir_a; _dir_a < main_dir_a + _config.MAX_DIRS;
         _dir_a++) {
      int dir_a = _dir_a % _config.MAX_DIRS;
      int dir_b =
          ((main_dir_b - main_dir_a + _config.MAX_DIRS) % _config.MAX_DIRS +
           _dir_a) %
          _config.MAX_DIRS;
      if (dir_a < frame_array_a.images.size() &&
          dir_b < frame_array_b.images.size()) {
        if (frame_array_b.images[dir_b].spLandmarkNum() > 0 &&
            frame_array_a.images[dir_a].spLandmarkNum() > 0) {
          dirs_a.push_back(dir_a);
          dirs_b.push_back(dir_b);
        }
      }
    }
  }

  Swarm::Pose extrinsic_a(frame_array_a.images[main_dir_a].extrinsic);
  Swarm::Pose extrinsic_b(frame_array_b.images[main_dir_b].extrinsic);
  Eigen::Quaterniond main_quat_new = extrinsic_a.att();
  Eigen::Quaterniond main_quat_old = extrinsic_b.att();

  int matched_dir_count = 0;
  for (size_t i = 0; i < dirs_a.size(); i++) {
    int dir_a = dirs_a[i];
    int dir_b = dirs_b[i];
    std::vector<Vector3d> _lm_norm_3d_b;
    std::vector<Vector3d> _lm_pos_a;
    std::vector<int> _idx_a;
    std::vector<int> _idx_b;
    std::vector<int> _camera_indices;

    if (dir_a < frame_array_a.images.size() &&
        dir_b < frame_array_b.images.size() && dir_a >= 0 && dir_b >= 0) {
      bool succ = computeCorrespondFeatures(
          frame_array_a.images[dir_a], frame_array_b.images[dir_b], _lm_pos_a,
          _idx_a, _lm_norm_3d_b, _idx_b, _camera_indices);
      SPDLOG_INFO(
          "[LoopDetector] computeCorrespondFeatures on camera_index {}:{} "
          "gives {} common features",
          dir_b, dir_a, _lm_pos_a.size());
      if (!succ) {
        continue;
      }
    }

    if (_lm_pos_a.size() >= _config.MIN_MATCH_PRE_DIR) {
      matched_dir_count++;
    }

    Swarm::Pose _extrinsic_a(frame_array_a.images[dir_a].extrinsic);
    Swarm::Pose _extrinsic_b(frame_array_b.images[dir_b].extrinsic);

    for (size_t id = 0; id < _lm_norm_3d_b.size(); id++) {
      index2dirindex_a.push_back(std::make_pair(dir_a, _idx_a[id]));
      index2dirindex_b.push_back(std::make_pair(dir_b, _idx_b[id]));
    }
    lm_pos_a.insert(lm_pos_a.end(), _lm_pos_a.begin(), _lm_pos_a.end());
    lm_norm_3d_b.insert(lm_norm_3d_b.end(), _lm_norm_3d_b.begin(),
                        _lm_norm_3d_b.end());
    cam_indices.insert(cam_indices.end(), _camera_indices.begin(),
                       _camera_indices.end());
  }

  if (lm_norm_3d_b.size() > _config.loop_inlier_feature_num &&
      matched_dir_count >= _config.MIN_DIRECTION_LOOP) {
    return true;
  } else {
    SPDLOG_WARN(
        "[LoopDetector::computeCorrImageArray@{}] Failed: features {}/{} dirs "
        "{}/{}",
        self_id, lm_norm_3d_b.size(), _config.loop_inlier_feature_num,
        matched_dir_count, _config.MIN_DIRECTION_LOOP);
    return false;
  }
}

bool LoopDetector::computeCorrespondFeatures(
    const VisualImageDesc &img_desc_a, const VisualImageDesc &img_desc_b,
    std::vector<Vector3d> &lm_pos_a, std::vector<int> &idx_a,
    std::vector<Vector3d> &lm_norm_3d_b, std::vector<int> &idx_b,
    std::vector<int> &cam_indices) {
  std::vector<cv::DMatch> _matches;
  auto &_a_lms = img_desc_a.landmarks;
  auto &_b_lms = img_desc_b.landmarks;

  if (_config.enable_superglue) {
    auto kpts_a = img_desc_a.landmarks2D(true, true);
    auto kpts_b = img_desc_b.landmarks2D(true, true);
    auto &desc0 = img_desc_a.landmark_descriptor;
    auto &desc1 = img_desc_b.landmark_descriptor;
    auto &scores0 = img_desc_a.landmark_scores;
    auto &scores1 = img_desc_b.landmark_scores;
    _matches =
        superglue->inference(kpts_a, kpts_b, desc0, desc1, scores0, scores1);
  } else {
    assert(img_desc_a.spLandmarkNum() * params->superpoint_dims ==
               img_desc_a.landmark_descriptor.size() &&
           "Desciptor size of new img desc must equal to to landmarks*256!!!");
    assert(img_desc_b.spLandmarkNum() * params->superpoint_dims ==
               img_desc_b.landmark_descriptor.size() &&
           "Desciptor size of old img desc must equal to to landmarks*256!!!");
    cv::Mat descriptors_a(img_desc_a.spLandmarkNum(), params->superpoint_dims,
                          CV_32F);
    memcpy(descriptors_a.data, img_desc_a.landmark_descriptor.data(),
           img_desc_a.landmark_descriptor.size() * sizeof(float));
    cv::Mat descriptors_b(img_desc_b.spLandmarkNum(), params->superpoint_dims,
                          CV_32F);
    memcpy(descriptors_b.data, img_desc_b.landmark_descriptor.data(),
           img_desc_b.landmark_descriptor.size() * sizeof(float));
    if (_config.enable_knn_match) {
      _matches =
          matchKNN(descriptors_a, descriptors_b, _config.knn_match_ratio);
    } else {
      cv::BFMatcher bfmatcher(cv::NORM_L2, true);
      bfmatcher.match(descriptors_a, descriptors_b, _matches);
    }
  }
  Point2fVector lm_b_2d, lm_a_2d;
  std::lock_guard<std::recursive_mutex> guard(landmark_mutex);
  for (auto match : _matches) {
    int index_a = match.queryIdx;
    int index_b = match.trainIdx;
    auto landmark_id = _a_lms[index_a].landmark_id;
    if (landmark_db.find(landmark_id) == landmark_db.end()) {
      continue;
    }
    if (landmark_db.at(landmark_id).flag == LandmarkFlag::UNINITIALIZED ||
        landmark_db.at(landmark_id).flag == LandmarkFlag::OUTLIER) {
      // ROS_WARN("Landmark %ld is not estimated or is outlier tracks %ld multi
      // %ld", landmark_id,
      //     landmark_db.at(landmark_id).track.size(),
      //     landmark_db.at(landmark_id).isMultiCamera());
      continue;
    }
    Vector3d pt3d_norm_b = _b_lms[index_b].pt3d_norm;
    lm_a_2d.push_back(_a_lms[index_a].pt2d);
    lm_b_2d.push_back(_b_lms[index_b].pt2d);
    idx_a.push_back(index_a);
    idx_b.push_back(index_b);
    lm_pos_a.push_back(landmark_db.at(landmark_id).position);
    lm_norm_3d_b.push_back(pt3d_norm_b);
    cam_indices.push_back(img_desc_b.camera_index);
  }

  if (lm_b_2d.size() < 4) {
    return false;
  }
  if (_config.enable_homography_test && !_config.enable_superglue) {
    std::vector<unsigned char> mask;
    cv::findHomography(lm_b_2d, lm_a_2d, cv::RANSAC, 10.0, mask);
    reduceVector(idx_a, mask);
    reduceVector(idx_b, mask);
    reduceVector(lm_pos_a, mask);
    reduceVector(lm_norm_3d_b, mask);
  }
  return true;
}

// Require 3d points of frame a and 2d point of frame b
bool LoopDetector::computeLoop(const VisualImageDescArray &frame_array_a,
                               const VisualImageDescArray &frame_array_b,
                               int main_dir_a, int main_dir_b, LoopEdge &ret) {
  if (frame_array_a.spLandmarkNum() < _config.loop_inlier_feature_num) {
    return false;
  }
  // Recover imformation

  assert(frame_array_a.drone_id == self_id &&
         "frame_array_a must from self drone to provide more 2d points!");

  bool success = false;

  double t_b = frame_array_b.stamp - t0;
  double t_a = frame_array_a.stamp - t0;
  printf(
      "[LoopDetector::computeLoop@%d] Compute loop drone b %d(d%d,dir %d)->a "
      "%d(d%d,dir %d) t %.1f->%.1f(%.1f)s landmarks %d:%d.\n",
      self_id, frame_array_b.frame_id, frame_array_b.drone_id, main_dir_b,
      frame_array_a.frame_id, frame_array_a.drone_id, main_dir_a, t_b, t_a,
      t_a - t_b, frame_array_b.spLandmarkNum(), frame_array_a.spLandmarkNum());

  std::vector<Vector3d> lm_pos_a;
  std::vector<Vector3d> lm_norm_3d_b;
  std::vector<int> dirs_a;
  Swarm::Pose DP_old_to_new;
  std::vector<int> inliers;
  std::vector<int> camera_indices;
  std::vector<std::pair<int, int>> index2dirindex_a, index2dirindex_b;

  success = computeCorrespondFeaturesOnImageArray(
      frame_array_a, frame_array_b, main_dir_a, main_dir_b, lm_pos_a,
      lm_norm_3d_b, camera_indices, index2dirindex_a, index2dirindex_b);

  if (success) {
    std::vector<Swarm::Pose> extrinsics;
    for (auto &img : frame_array_b.images) {
      extrinsics.push_back(img.extrinsic);
    }
    success = computeRelativePosePnPnonCentral(
        lm_pos_a, lm_norm_3d_b, extrinsics, camera_indices,
        frame_array_a.pose_drone, frame_array_b.pose_drone, DP_old_to_new,
        inliers, _config.is_4dof);
    if (success) {
      // setup return loop
      ret.relative_pose = DP_old_to_new.toROS();
      ret.drone_id_a = frame_array_b.drone_id;
      ret.ts_a = ros::Time(frame_array_b.stamp);

      ret.drone_id_b = frame_array_a.drone_id;
      ret.ts_b = ros::Time(frame_array_a.stamp);

      ret.self_pose_a = toROSPose(frame_array_b.pose_drone);
      ret.self_pose_b = toROSPose(frame_array_a.pose_drone);

      ret.keyframe_id_a = frame_array_b.frame_id;
      ret.keyframe_id_b = frame_array_a.frame_id;

      ret.pos_cov.x = _config.loop_cov_pos;
      ret.pos_cov.y = _config.loop_cov_pos;
      ret.pos_cov.z = _config.loop_cov_pos;

      ret.ang_cov.x = _config.loop_cov_ang;
      ret.ang_cov.y = _config.loop_cov_ang;
      ret.ang_cov.z = _config.loop_cov_ang;

      ret.pnp_inlier_num = inliers.size();
      ret.id = self_id * MAX_LOOP_ID + loop_count;

      if (checkLoopOdometryConsistency(ret)) {
        loop_count++;
        SPDLOG_INFO(
            "[LoopDetector] Loop {} Detected {}->{} dt {:.3f}s DPose {} "
            "inliers {}. Will publish\n",
            ret.id, ret.drone_id_a, ret.drone_id_b,
            (ret.ts_b - ret.ts_a).toSec(), DP_old_to_new.toStr().c_str(),
            ret.pnp_inlier_num);

        int new_d_id = frame_array_a.drone_id;
        int old_d_id = frame_array_b.drone_id;
        inter_drone_loop_count[new_d_id][old_d_id] =
            inter_drone_loop_count[new_d_id][old_d_id] + 1;
        inter_drone_loop_count[old_d_id][new_d_id] =
            inter_drone_loop_count[old_d_id][new_d_id] + 1;
      } else {
        success = false;
        SPDLOG_INFO(
            "[LoopDetector] Loop not consistency with odometry, give up.\n");
      }
    }
  }

  if (params->show) {
    drawMatched(frame_array_a, frame_array_b, main_dir_a, main_dir_b, success,
                inliers, DP_old_to_new, index2dirindex_a, index2dirindex_b);
  }

  return success;
}

void LoopDetector::drawMatched(
    const VisualImageDescArray &frame_array_a,
    const VisualImageDescArray &frame_array_b, int main_dir_a, int main_dir_b,
    bool success, std::vector<int> inliers, Swarm::Pose DP_b_to_a,
    std::vector<std::pair<int, int>> index2dirindex_a,
    std::vector<std::pair<int, int>> index2dirindex_b) {
  cv::Mat show;
  char title[100] = {0};
  std::vector<cv::Mat> _matched_imgs;
  auto &imgs_a = msgid2cvimgs[frame_array_a.frame_id];
  auto &imgs_b = msgid2cvimgs[frame_array_b.frame_id];
  _matched_imgs.resize(imgs_b.size());
  for (size_t i = 0; i < imgs_b.size(); i++) {
    int dir_a =
        ((-main_dir_b + main_dir_a + _config.MAX_DIRS) % _config.MAX_DIRS + i) %
        _config.MAX_DIRS;
    if (!imgs_b[i].empty() && !imgs_a[dir_a].empty()) {
      cv::vconcat(imgs_b[i], imgs_a[dir_a], _matched_imgs[i]);
      if (_matched_imgs[i].channels() != 3) {
        cv::cvtColor(_matched_imgs[i], _matched_imgs[i], cv::COLOR_GRAY2BGR);
      }
    }
  }
  std::set<int> inlier_set(inliers.begin(), inliers.end());

  for (unsigned int i = 0; i < index2dirindex_a.size(); i++) {
    int old_pt_id = index2dirindex_b[i].second;
    int old_dir_id = index2dirindex_b[i].first;

    int new_pt_id = index2dirindex_a[i].second;
    int new_dir_id = index2dirindex_a[i].first;
    auto pt_old = frame_array_b.images[old_dir_id].landmarks[old_pt_id].pt2d;
    auto pt_new = frame_array_a.images[new_dir_id].landmarks[new_pt_id].pt2d;
    if (_matched_imgs[old_dir_id].empty()) {
      continue;
    }
    cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
    int line_thickness = 2;
    if (inlier_set.find(i) == inlier_set.end()) {
      color = cv::Scalar(0, 0, 0);
      line_thickness = 1;
    }
    cv::line(_matched_imgs[old_dir_id], pt_old,
             pt_new + cv::Point2f(0, imgs_b[old_dir_id].rows), color,
             line_thickness);
    cv::circle(_matched_imgs[old_dir_id], pt_old, 3, color, 1);
    cv::circle(_matched_imgs[new_dir_id],
               pt_new + cv::Point2f(0, imgs_b[old_dir_id].rows), 3, color, 1);
  }

  show = _matched_imgs[0];
  for (size_t i = 1; i < _matched_imgs.size(); i++) {
    if (_matched_imgs[i].empty()) continue;
    cv::line(_matched_imgs[i], cv::Point2f(0, 0),
             cv::Point2f(0, _matched_imgs[i].rows), cv::Scalar(255, 255, 0), 2);
    cv::hconcat(show, _matched_imgs[i], show);
  }

  double dt = (frame_array_a.stamp - frame_array_b.stamp);
  if (success) {
    auto ypr = DP_b_to_a.rpy() * 180 / M_PI;
    sprintf(title, "Loop: %d->%d dt %3.1fs LM %d/%d", frame_array_b.drone_id,
            frame_array_a.drone_id, dt, inliers.size(),
            index2dirindex_a.size());
    cv::putText(show, title, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);

    sprintf(title, "%s", DP_b_to_a.toStr().c_str());
    cv::putText(show, title, cv::Point2f(20, 60), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);
    sprintf(title, "%d<->%d", frame_array_b.frame_id, frame_array_a.frame_id);
    cv::putText(show, title, cv::Point2f(20, 90), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);
    sprintf(title, "Ego A: %s", frame_array_a.pose_drone.toStr().c_str());
    cv::putText(show, title, cv::Point2f(20, 120), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);
    sprintf(title, "Ego B: %s", frame_array_b.pose_drone.toStr().c_str());
    cv::putText(show, title, cv::Point2f(20, 150), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 255, 0), 2);

  } else {
    sprintf(title, "FAILED %d->%d dt %3.1fs LM %d/%d", frame_array_b.drone_id,
            frame_array_a.drone_id, dt, inliers.size(),
            index2dirindex_a.size());
    cv::putText(show, title, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(0, 0, 255), 2);
  }

  // cv::resize(show, show, cv::Size(), 2, 2);
  static int loop_match_count = 0;
  loop_match_count++;
  char PATH[100] = {0};

  if (!show.empty() && success) {
    sprintf(PATH, "loop/match%d.png", loop_match_count);
    cv::imwrite(params->OUTPUT_PATH + PATH, show);
  }
  cv::imshow("Matches", show);
  cv::waitKey(10);
}
void LoopDetector::onLoopConnection(LoopEdge &loop_conn) {
  on_loop_cb(loop_conn);
}

void LoopDetector::updatebyLandmarkDB(
    const std::map<LandmarkIdType, LandmarkPerId> &vins_landmark_db) {
  std::lock_guard<std::recursive_mutex> guard(landmark_mutex);
  for (auto it : vins_landmark_db) {
    auto landmark_id = it.first;
    if (landmark_db.find(landmark_id) == landmark_db.end()) {
      landmark_db[landmark_id] = it.second;
    } else {
      if (it.second.flag == LandmarkFlag::INITIALIZED ||
          it.second.flag == LandmarkFlag::ESTIMATED) {
        landmark_db[landmark_id] = it.second;
      }
    }
  }
}

void LoopDetector::updatebySldWin(const std::vector<VINSFramePtr> sld_win) {
  const std::lock_guard<std::mutex> lock(keyframe_database_mutex);
  for (auto frame : sld_win) {
    auto frame_id = frame->frame_id;
    if (keyframe_database.find(frame_id) != keyframe_database.end()) {
      keyframe_database.at(frame_id).pose_drone = frame->odom.pose();
    }
  }
}

bool LoopDetector::hasFrame(FrameIdType frame_id) {
  const std::lock_guard<std::mutex> lock(keyframe_database_mutex);
  return keyframe_database.find(frame_id) != keyframe_database.end();
}

LoopDetector::LoopDetector(int _self_id, const LoopDetectorConfig &config)
    : self_id(_self_id),
      _config(config),
      local_index(params->netvlad_dims),
      remote_index(params->netvlad_dims),
      ego_motion_traj(_self_id, true, _config.pos_covariance_per_meter,
                      _config.yaw_covariance_per_meter) {
  if (_config.enable_superglue) {
    superglue = new SuperGlueOnnx(_config.superglue_model_path);
  }
}

}  // namespace D2FrontEnd