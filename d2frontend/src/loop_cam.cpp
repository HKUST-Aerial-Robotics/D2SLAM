#include <camodocal/camera_models/CameraFactory.h>

#include <memory>

#include <cv_bridge/cv_bridge.h>
#include "opencv2/features2d.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>


#include <swarm_msgs/swarm_lcm_converter.hpp>
#include <d2common/fisheye_undistort.h>
#include <d2frontend/d2featuretracker.h>
#include <d2frontend/loop_cam.h>
#include "d2frontend/CNN/superpoint_onnx.h"

using namespace std::chrono;

double TRIANGLE_THRES;

namespace D2FrontEnd {
LoopCam::LoopCam(LoopCamConfig config, ros::NodeHandle &nh)
    : camera_configuration(config.camera_configuration),
      self_id(config.self_id),
      _config(config) {
  int img_width =
      config.enable_undistort_image ? params->width_undistort : params->width;
  int img_height =
      config.enable_undistort_image ? params->height_undistort : params->height;

  if (config.cnn_use_onnx) {
    SPDLOG_INFO("Init CNNs using onnx");
    netvlad_onnx = new MobileNetVLADONNX(
        config.netvlad_model, img_width, img_height, config.cnn_enable_tensorrt,
        config.cnn_enable_tensorrt_fp16, config.cnn_enable_tensorrt_int8,
        config.netvlad_int8_calib_table_name);
        
    SuperPointConfig sp_config = config.superpoint_config;

#ifdef USE_CUDA
    superpoint_ptr = std::make_unique<SuperPoint>(sp_config);
    if (superpoint_ptr->build()){
      SPDLOG_INFO("SuperPoint build success");
    } else {
      SPDLOG_ERROR("SuperPoint build failed");
    }
#else
    superpoint_ptr = new SuperPointONNX(
        config.superpoint_model, ((int)(params->feature_min_dist / 2)),
        config.pca_comp, config.pca_mean, img_width, img_height,
        config.superpoint_thres, config.superpoint_max_num,
        config.cnn_enable_tensorrt, config.cnn_enable_tensorrt_fp16,
        config.cnn_enable_tensorrt_int8,
        config.superpoint_int8_calib_table_name);
#endif    

  }
  undistortors = params->undistortors;
  cams = params->camera_ptrs;
  SPDLOG_INFO("Deepnet ready");
  if (_config.OUTPUT_RAW_SUPERPOINT_DESC) {
    fsp.open(params->OUTPUT_PATH + "superpoint.csv", std::fstream::app);
  }
}

void LoopCam::encodeImage(const cv::Mat &_img, VisualImageDesc &_img_desc) {
  std::vector<int> jpg_params;
  jpg_params.push_back(cv::IMWRITE_JPEG_QUALITY);
  jpg_params.push_back(params->JPG_QUALITY);

  cv::imencode(".jpg", _img, _img_desc.image, jpg_params);
  _img_desc.image_width = _img.cols;
  _img_desc.image_width = _img.rows;
  // std::cout << "IMENCODE Cost " <<
  // duration_cast<microseconds>(high_resolution_clock::now() -
  // start).count()/1000.0 << "ms" << std::endl; std::cout << "JPG SIZE" <<
  // _img_desc.image.size() << std::endl;
}

double triangulatePoint(Eigen::Quaterniond q0, Eigen::Vector3d t0,
                        Eigen::Quaterniond q1, Eigen::Vector3d t1,
                        Eigen::Vector2d point0, Eigen::Vector2d point1,
                        Eigen::Vector3d &point_3d) {
  Eigen::Matrix3d R0 = q0.toRotationMatrix();
  Eigen::Matrix3d R1 = q1.toRotationMatrix();

  // std::cout << "RO" << R0 << "T0" << t0.transpose() << std::endl;
  // std::cout << "R1" << R1 << "T1" << t1.transpose() << std::endl;

  Eigen::Matrix<double, 3, 4> Pose0;
  Pose0.leftCols<3>() = R0.transpose();
  Pose0.rightCols<1>() = -R0.transpose() * t0;

  Eigen::Matrix<double, 3, 4> Pose1;
  Pose1.leftCols<3>() = R1.transpose();
  Pose1.rightCols<1>() = -R1.transpose() * t1;

  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
  Eigen::Vector4d triangulated_point;
  triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);

  Eigen::MatrixXd pts(4, 1);
  pts << point_3d.x(), point_3d.y(), point_3d.z(), 1;
  Eigen::MatrixXd errs = design_matrix * pts;
  return errs.norm() / errs.rows();
}

template <typename T>
void reduceVector(std::vector<T> &v, std::vector<uchar> status) {
  int j = 0;
  for (unsigned int i = 0; i < v.size(); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

cv::Mat drawMatches(std::vector<cv::Point2f> pts1,
                    std::vector<cv::Point2f> pts2,
                    std::vector<cv::DMatch> _matches, const cv::Mat &up,
                    const cv::Mat &down) {
  std::vector<cv::KeyPoint> kps1;
  std::vector<cv::KeyPoint> kps2;

  for (auto pt : pts1) {
    cv::KeyPoint kp;
    kp.pt = pt;
    kps1.push_back(kp);
  }

  for (auto pt : pts2) {
    cv::KeyPoint kp;
    kp.pt = pt;
    kps2.push_back(kp);
  }

  cv::Mat _show;

  cv::drawMatches(up, kps1, down, kps2, _matches, _show);

  return _show;
}

void matchLocalFeatures(std::vector<cv::Point2f> &pts_up,
                        std::vector<cv::Point2f> &pts_down,
                        std::vector<float> &_desc_up,
                        std::vector<float> &_desc_down,
                        std::vector<int> &ids_up, std::vector<int> &ids_down) {
  // printf("matchLocalFeatures %ld %ld: ", pts_up.size(), pts_down.size());
  const cv::Mat desc_up(_desc_up.size() / params->superpoint_dims,
                        params->superpoint_dims, CV_32F, _desc_up.data());
  const cv::Mat desc_down(_desc_down.size() / params->superpoint_dims,
                          params->superpoint_dims, CV_32F, _desc_down.data());

  cv::BFMatcher bfmatcher(cv::NORM_L2, true);

  std::vector<cv::DMatch> _matches;
  bfmatcher.match(desc_up, desc_down, _matches);

  std::vector<cv::Point2f> _pts_up, _pts_down;
  std::vector<int> ids;
  for (auto match : _matches) {
    if (match.distance < ACCEPT_SP_MATCH_DISTANCE || true) {
      int now_id = match.queryIdx;
      int old_id = match.trainIdx;
      _pts_up.push_back(pts_up[now_id]);
      _pts_down.push_back(pts_down[old_id]);
      ids_up.push_back(now_id);
      ids_down.push_back(old_id);
    } else {
      std::cout << "Giveup match dis" << match.distance << std::endl;
    }
  }

  std::vector<uint8_t> status;

  pts_up = std::vector<cv::Point2f>(_pts_up);
  pts_down = std::vector<cv::Point2f>(_pts_down);
}

VisualImageDescArray LoopCam::processStereoframe(const StereoFrame &msg) {
  VisualImageDescArray visual_array;
  visual_array.stamp = msg.stamp.toSec();

  cv::Mat _show, tmp;
  TicToc tt;
  static int t_count = 0;
  static double tt_sum = 0;

  if (camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
    visual_array.images.resize(4);
  }

  for (unsigned int i = 0; i < msg.left_images.size(); i++) {
    if (camera_configuration == CameraConfig::PINHOLE_DEPTH) {
      visual_array.images.push_back(
          generateGrayDepthImageDescriptor(msg, i, tmp));
    } else if (camera_configuration == CameraConfig::STEREO_PINHOLE) {
      auto _imgs = generateStereoImageDescriptor(msg, i, tmp);
      if (_config.stereo_as_depth_cam) {
        if (_config.right_cam_as_main) {
          visual_array.images.push_back(_imgs[1]);
        } else {
          visual_array.images.push_back(_imgs[0]);
        }
      } else {
        visual_array.images.push_back(_imgs[0]);
        visual_array.images.push_back(_imgs[1]);
      }
    } else if (camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
      auto seq = params->camera_seq[i];
      visual_array.images[seq] = generateImageDescriptor(msg, i, tmp);
    }

    if (_show.cols == 0) {
      _show = tmp;
    } else {
      cv::hconcat(_show, tmp, _show);
    }
  }

  tt_sum += tt.toc();
  t_count += 1;
  spdlog::debug("KF Count {} loop_cam cost avg {:.1f}ms cur {:.1f}ms", kf_count,
                tt_sum / t_count, tt.toc());

  visual_array.frame_id = msg.keyframe_id;
  visual_array.pose_drone = msg.pose_drone;
  visual_array.drone_id = self_id;

  if (_config.show && !_show.empty()) {
    char text[100] = {0};
    char PATH[100] = {0};
    sprintf(text, "FEATURES@Drone%d", self_id);
    sprintf(PATH, "loop/features%d.png", kf_count);
    cv::imshow(text, _show);
    cv::imwrite(params->OUTPUT_PATH + PATH, _show);
    cv::waitKey(10);
  }
  kf_count++;
  visual_array.sync_landmark_ids();
  return visual_array;
}

VisualImageDesc LoopCam::generateImageDescriptor(const StereoFrame &msg,
                                                 int vcam_id, cv::Mat &_show) {
  if (vcam_id > msg.left_images.size()) {
    SPDLOG_WARN("Flatten images too few");
    VisualImageDesc ides;
    ides.stamp = msg.stamp.toSec();
    return ides;
  }
  cv::Mat undist = msg.left_images[vcam_id];
  TicToc tt;
  if (_config.enable_undistort_image) {
#ifdef USE_CUDA
    undist = cv::Mat(undistortors[vcam_id]->undist_id_cuda(undist, 0, true));
#else
    SPDLOG_WARN("Undistort image not supported without CUDA");
#endif
  }
  if (params->enable_perf_output) {
    SPDLOG_INFO("[D2Frontend::LoopCam] undist image cost {:.1f}ms", tt.toc());
  }
  VisualImageDesc vframe = extractorImgDescDeepnet(
      msg.stamp, undist, msg.left_camera_indices[vcam_id],
      msg.left_camera_ids[vcam_id], false);

  if (vframe.image_desc.size() == 0) {
    SPDLOG_WARN("Failed on deepnet: vframe.image_desc.size() == 0.");
    cv::Mat _img;
    // return ides;
  }

  auto start = high_resolution_clock::now();
  // std::cout << "Downsample and encode Cost " <<
  // duration_cast<microseconds>(high_resolution_clock::now() -
  // start).count()/1000.0 << "ms" << std::endl;

  vframe.extrinsic = msg.left_extrisincs[vcam_id];
  vframe.pose_drone = msg.pose_drone;
  vframe.frame_id = msg.keyframe_id;
  if (params->debug_plot_superpoint_features ||
      params->ftconfig->enable_lk_optical_flow || params->show) {
    vframe.raw_image = undist;
  }

  auto image_left = undist;
  auto pts_up = vframe.landmarks2D();
  std::vector<int> ids_up, ids_down;
  if (_config.send_img) {
    encodeImage(image_left, vframe);
  }
  if (_config.show) {
    cv::Mat img_up = image_left;
    if (!_config.send_img) {
      encodeImage(img_up, vframe);
    }
    cv::cvtColor(img_up, img_up, cv::COLOR_GRAY2BGR);
    for (unsigned int i = 0; i < vframe.landmarkNum(); i++) {
      if (vframe.landmarks[i].depth_mea) {
        auto pt = vframe.landmarks[i].pt2d;
        auto dep = vframe.landmarks[i].depth;
        cv::circle(img_up, pt, 3, cv::Scalar(0, 255, 0), 1);
        char idtext[100] = {};
        sprintf(idtext, "%3.2f", dep);
        cv::putText(img_up, idtext, pt - cv::Point2f(5, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
      }
    }
    _show = img_up;
    char text[100] = {0};
    sprintf(text, "Frame %d: %ld Features %d", kf_count, msg.keyframe_id,
            pts_up.size());
    cv::putText(_show, text, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1.5);
  }
  return vframe;
}

VisualImageDesc LoopCam::generateGrayDepthImageDescriptor(
    const StereoFrame &msg, int vcam_id, cv::Mat &_show) {
  if (vcam_id > msg.left_images.size()) {
    SPDLOG_WARN("Flatten images too few");
    VisualImageDesc ides;
    ides.stamp = msg.stamp.toSec();
    return ides;
  }

  VisualImageDesc vframe = extractorImgDescDeepnet(
      msg.stamp, msg.left_images[vcam_id], msg.left_camera_indices[vcam_id],
      msg.left_camera_ids[vcam_id], false);

  if (vframe.image_desc.size() == 0) {
    SPDLOG_WARN("Failed on deepnet: vframe.image_desc.size() == 0.");
    cv::Mat _img;
    // return ides;
  }

  auto start = high_resolution_clock::now();
  // std::cout << "Downsample and encode Cost " <<
  // duration_cast<microseconds>(high_resolution_clock::now() -
  // start).count()/1000.0 << "ms" << std::endl;

  vframe.extrinsic = msg.left_extrisincs[vcam_id];
  vframe.pose_drone = msg.pose_drone;
  vframe.frame_id = msg.keyframe_id;
  if (params->debug_plot_superpoint_features ||
      params->ftconfig->enable_lk_optical_flow || params->show) {
    vframe.raw_image = msg.left_images[vcam_id];
    vframe.raw_depth_image = msg.depth_images[vcam_id];
  }

  auto image_left = msg.left_images[vcam_id];
  auto pts_up = vframe.landmarks2D();
  std::vector<int> ids_up, ids_down;
  Swarm::Pose pose_drone(msg.pose_drone);
  Swarm::Pose pose_cam = pose_drone * Swarm::Pose(msg.left_extrisincs[vcam_id]);

  std::vector<float> desc_new;

  int count_3d = 0;
  for (unsigned int i = 0; i < pts_up.size(); i++) {
    cv::Point2f pt_up = pts_up[i];
    if (pt_up.x < 0 || pt_up.x >= params->width || pt_up.y < 0 ||
        pt_up.y >= params->height) {
      continue;
    }

    auto dep = msg.depth_images[vcam_id].at<unsigned short>(pt_up) / 1000.0;
    if (dep > _config.DEPTH_NEAR_THRES && dep < _config.DEPTH_FAR_THRES) {
      Eigen::Vector3d pt_up3d, pt_down3d;
      cams.at(vcam_id)->liftProjective(Vector2d(pt_up.x, pt_up.y), pt_up3d);

      Eigen::Vector3d pt2d_norm(pt_up3d.x() / pt_up3d.z(),
                                pt_up3d.y() / pt_up3d.z(), 1);
      auto pt3dcam = pt2d_norm * dep;
      Eigen::Vector3d pt3d = pose_cam * pt3dcam;
      // printf("landmark raw depth %f pt3dcam %f %f %f pt2d_norm %f %f distance
      // %f\n", dep,
      //     pt3dcam.x(), pt3dcam.y(), pt3dcam.z(), pt2d_norm.x(),
      //     pt2d_norm.y(), pt3dcam.norm());

      vframe.landmarks[i].pt3d = pt3d;
      vframe.landmarks[i].flag = LandmarkFlag::UNINITIALIZED;
      vframe.landmarks[i].depth = pt3dcam.norm();
      vframe.landmarks[i].depth_mea = true;
      count_3d++;
    }
  }

  // ROS_INFO("Image 2d kpts: %ld 3d : %d desc size %ld",
  // ides.landmarks_2d.size(), count_3d, ides.feature_descriptor.size());

  if (_config.send_img) {
    encodeImage(image_left, vframe);
  }

  if (_config.show) {
    cv::Mat img_up = image_left;

    if (!_config.send_img) {
      encodeImage(img_up, vframe);
    }

    cv::cvtColor(img_up, img_up, cv::COLOR_GRAY2BGR);

    for (unsigned int i = 0; i < vframe.landmarkNum(); i++) {
      if (vframe.landmarks[i].depth_mea) {
        auto pt = vframe.landmarks[i].pt2d;
        auto dep = vframe.landmarks[i].depth;
        cv::circle(img_up, pt, 3, cv::Scalar(0, 255, 0), 1);
        char idtext[100] = {};
        sprintf(idtext, "%3.2f", dep);
        cv::putText(img_up, idtext, pt - cv::Point2f(5, 0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
      }
    }

    _show = img_up;
    char text[100] = {0};
    sprintf(text, "Frame %d: %ld Features %d/%d", kf_count, msg.keyframe_id,
            count_3d, pts_up.size());
    cv::putText(_show, text, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1.5);
  }

  return vframe;
}

std::vector<VisualImageDesc> LoopCam::generateStereoImageDescriptor(
    const StereoFrame &msg, int vcam_id, cv::Mat &_show) {
  // This function currently only support pinhole-like stereo camera.
  auto vframe0 = extractorImgDescDeepnet(
      msg.stamp, msg.left_images[vcam_id], msg.left_camera_indices[vcam_id],
      msg.left_camera_ids[vcam_id], _config.right_cam_as_main);
  auto vframe1 = extractorImgDescDeepnet(
      msg.stamp, msg.right_images[vcam_id], msg.right_camera_indices[vcam_id],
      msg.right_camera_ids[vcam_id], !_config.right_cam_as_main);

  if (vframe0.image_desc.size() == 0 && vframe1.image_desc.size() == 0) {
    SPDLOG_WARN(
        "Failed on deepnet: vframe0.image_desc.size() == 0 && "
        "vframe1.image_desc.size() == 0");
    // cv::Mat _img;
    // return ides;
  }

  auto start = high_resolution_clock::now();
  // std::cout << "Downsample and encode Cost " <<
  // duration_cast<microseconds>(high_resolution_clock::now() -
  // start).count()/1000.0 << "ms" << std::endl;

  vframe0.extrinsic = msg.left_extrisincs[vcam_id];
  vframe0.pose_drone = msg.pose_drone;
  vframe0.frame_id = msg.keyframe_id;

  vframe1.extrinsic = msg.right_extrisincs[vcam_id];
  vframe1.pose_drone = msg.pose_drone;
  vframe1.frame_id = msg.keyframe_id;

  auto image_left = msg.left_images[vcam_id];
  auto image_right = msg.right_images[vcam_id];

  auto pts_up = vframe0.landmarks2D();
  auto pts_down = vframe1.landmarks2D();
  std::vector<int> ids_up, ids_down;

  int count_3d = 0;
  if (_config.stereo_as_depth_cam) {
    if (vframe0.landmarkNum() > _config.ACCEPT_MIN_3D_PTS) {
      matchLocalFeatures(pts_up, pts_down, vframe0.landmark_descriptor,
                         vframe1.landmark_descriptor, ids_up, ids_down);
    }
    Swarm::Pose pose_drone(msg.pose_drone);
    Swarm::Pose pose_up =
        pose_drone * Swarm::Pose(msg.left_extrisincs[vcam_id]);
    Swarm::Pose pose_down =
        pose_drone * Swarm::Pose(msg.right_extrisincs[vcam_id]);
    for (unsigned int i = 0; i < pts_up.size(); i++) {
      auto pt_up = pts_up[i];
      auto pt_down = pts_down[i];

      Eigen::Vector3d pt_up3d, pt_down3d;
      // TODO: This may not work for stereo fisheye. Pending to update.
      cams.at(msg.left_camera_indices[vcam_id])
          ->liftProjective(Eigen::Vector2d(pt_up.x, pt_up.y), pt_up3d);
      cams.at(msg.right_camera_indices[vcam_id])
          ->liftProjective(Eigen::Vector2d(pt_down.x, pt_down.y), pt_down3d);

      Eigen::Vector2d pt_up_norm(pt_up3d.x() / pt_up3d.z(),
                                 pt_up3d.y() / pt_up3d.z());
      Eigen::Vector2d pt_down_norm(pt_down3d.x() / pt_down3d.z(),
                                   pt_down3d.y() / pt_down3d.z());

      Eigen::Vector3d point_3d;
      double err =
          triangulatePoint(pose_up.att(), pose_up.pos(), pose_down.att(),
                           pose_down.pos(), pt_up_norm, pt_down_norm, point_3d);

      auto pt_cam = pose_up.att().inverse() * (point_3d - pose_up.pos());

      if (err > TRIANGLE_THRES) {
        continue;
      }

      int idx = ids_up[i];
      int idx_down = ids_down[i];
      vframe0.landmarks[idx].pt3d = point_3d;
      vframe0.landmarks[idx].depth_mea = true;
      vframe0.landmarks[idx].depth = pt_cam.norm();
      vframe0.landmarks[idx].flag = LandmarkFlag::UNINITIALIZED;

      auto pt_cam2 = pose_down.att().inverse() * (point_3d - pose_down.pos());
      vframe1.landmarks[idx_down].pt3d = point_3d;
      vframe0.landmarks[idx].depth_mea = true;
      vframe0.landmarks[idx].depth = pt_cam2.norm();
      vframe1.landmarks[idx_down].flag = LandmarkFlag::UNINITIALIZED;
      count_3d++;
    }
  }

  if (_config.send_img) {
    encodeImage(image_left, vframe0);
    encodeImage(image_right, vframe1);
  }

  if (_config.show) {
    cv::Mat img_up = image_left;
    cv::Mat img_down = image_right;
    if (!_config.send_img) {
      encodeImage(img_up, vframe0);
      encodeImage(img_down, vframe1);
    }
    cv::cvtColor(img_up, img_up, cv::COLOR_GRAY2BGR);
    cv::cvtColor(img_down, img_down, cv::COLOR_GRAY2BGR);

    for (auto pt : pts_down) {
      cv::circle(img_down, pt, 1, cv::Scalar(255, 0, 0), -1);
    }

    for (auto _pt : vframe0.landmarks2D()) {
      cv::Point2f pt(_pt.x, _pt.y);
      cv::circle(img_up, pt, 3, cv::Scalar(0, 0, 255), 1);
    }

    cv::vconcat(img_up, img_down, _show);
    if (_config.stereo_as_depth_cam) {
      for (unsigned int i = 0; i < pts_up.size(); i++) {
        int idx = ids_up[i];
        if (vframe0.landmarks[idx].flag) {
          char title[100] = {0};
          auto pt = pts_up[i];
          cv::circle(_show, pt, 3, cv::Scalar(0, 255, 0), 1);
          cv::arrowedLine(_show, pts_up[i], pts_down[i],
                          cv::Scalar(255, 255, 0), 1);
        }
      }
    }

    char text[100] = {0};
    sprintf(text, "Frame %ld Features %d/%d", msg.keyframe_id, count_3d,
            pts_up.size());
    cv::putText(_show, text, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1.5);
  }

  if (params->debug_plot_superpoint_features || params->show ||
      params->ftconfig->enable_lk_optical_flow) {
    vframe1.raw_image = msg.right_images[vcam_id];
    vframe0.raw_image = msg.left_images[vcam_id];
  }
  std::vector<VisualImageDesc> ret{vframe0, vframe1};
  return ret;
}

VisualImageDesc LoopCam::extractorImgDescDeepnet(ros::Time stamp, cv::Mat img,
                                                 int camera_index,
                                                 int camera_id,
                                                 bool superpoint_mode) {
  auto start = high_resolution_clock::now();

  VisualImageDesc vframe;
  vframe.stamp = stamp.toSec();
  vframe.camera_index = camera_index;
  vframe.camera_id = camera_id;
  vframe.drone_id = self_id;

  if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
    cv::Mat roi = img(cv::Rect(0, img.rows * 3 / 4, img.cols, img.rows / 4));
    roi.setTo(cv::Scalar(0, 0, 0));
  }
  std::vector<cv::Point2f> landmarks_2d;
  if (_config.superpoint_max_num > 0) {
    // We only inference when superpoint max num > 0
    // otherwise, d2vins only uses LK optical flow feature.
    superpoint_ptr->infer(img, landmarks_2d, vframe.landmark_descriptor,
                                vframe.landmark_scores);
  }

  if (!superpoint_mode) {
    if (_config.cnn_use_onnx) {
      vframe.image_desc = netvlad_onnx->inference(img);
    }
  }

  for (unsigned int i = 0; i < landmarks_2d.size(); i++) {
    auto pt_up = landmarks_2d[i];
    Eigen::Vector3d pt_up3d;
    cams.at(camera_index)
        ->liftProjective(Eigen::Vector2d(pt_up.x, pt_up.y), pt_up3d);
    LandmarkPerFrame lm;
    lm.pt2d = pt_up;
    pt_up3d.normalize();
    if (pt_up3d.hasNaN()) {
      SPDLOG_WARN("NaN detected!!! This will inference landmark_descriptor\n");
      continue;
    }
    lm.pt3d_norm = pt_up3d;
    lm.camera_index = camera_index;
    lm.camera_id = camera_id;
    lm.stamp = vframe.stamp;
    lm.stamp_discover = vframe.stamp;
    lm.color = extractColor(img, pt_up);
    vframe.landmarks.emplace_back(lm);
    if (_config.OUTPUT_RAW_SUPERPOINT_DESC) {
      for (int j = 0; j < params->superpoint_dims; j++) {
        fsp << vframe.landmark_descriptor[i * params->superpoint_dims + j]
            << " ";
      }
      fsp << std::endl;
    }
  }

  return vframe;
}
}  // namespace D2FrontEnd