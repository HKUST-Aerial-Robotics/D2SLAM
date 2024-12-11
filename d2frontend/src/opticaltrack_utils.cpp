#include <d2common/d2basetypes.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/loop_detector.h>
#include <d2frontend/opticaltrack_utils.h>
#include <spdlog/spdlog.h>
#include <d2frontend/utils.h>

#include <d2common/utils.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#ifdef USE_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>
#else
#include <opencv2/optflow.hpp>
#endif

using namespace std::chrono;
using namespace D2Common;
using D2Common::Utility::TicToc;

#define WIN_SIZE cv::Size(21, 21)

namespace D2FrontEnd {

std::vector<cv::Point2f> detectFastByRegion(cv::InputArray _img,
                                            cv::InputArray _mask, int features,
                                            int cols, int rows);
                                            
bool checkPointsWithinBounds(const std::vector<cv::Point2f>& pts, const cv::Size& img_size, const cv::Size& win_size, std::vector<cv::Point2f>& filtered_pts);

bool inBorder(const cv::Point2f &pt, cv::Size shape) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < shape.width - BORDER_SIZE &&
         BORDER_SIZE <= img_y && img_y < shape.height - BORDER_SIZE;
}


std::vector<cv::Point2f> opticalflowTrack(const cv::Mat &cur_img,
                                          const cv::Mat &prev_img,
                                          std::vector<cv::Point2f> &prev_pts,
                                          std::vector<LandmarkIdType> &ids,
                                          TrackLRType type, bool enable_cuda) {
  if (prev_pts.size() == 0) {
    return std::vector<cv::Point2f>();
  }
  TicToc tic;
  std::vector<uchar> status;
  std::vector<cv::Point2f> cur_pts;
  float move_cols =
      cur_img.cols * 90.0 /
      params->undistort_fov;  // slightly lower than 0.5 cols when fov=200

  if (prev_pts.size() == 0) {
    return std::vector<cv::Point2f>();
  }

  if (type == WHOLE_IMG_MATCH) {
    cur_pts = prev_pts;
  } else {
    status.resize(prev_pts.size());
    std::fill(status.begin(), status.end(), 0);
    if (type == LEFT_RIGHT_IMG_MATCH) {
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
  std::vector<uchar> reverse_status;
  std::vector<cv::Point2f> reverse_pts;
  if (enable_cuda) {
#ifdef USE_CUDA
    cv::cuda::GpuMat gpu_prev_img(prev_img);
    cv::cuda::GpuMat gpu_cur_img(cur_img);
    cv::cuda::GpuMat gpu_prev_pts(prev_pts);
    cv::cuda::GpuMat gpu_cur_pts(cur_pts);
    cv::cuda::GpuMat gpu_status;
    cv::cuda::GpuMat reverse_gpu_status;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse =
        cv::cuda::SparsePyrLKOpticalFlow::create(WIN_SIZE, PYR_LEVEL, 30, true);
    d_pyrLK_sparse->calc(gpu_prev_img, gpu_cur_img, gpu_prev_pts, gpu_cur_pts,
                         gpu_status);
    gpu_status.download(status);
    gpu_cur_pts.download(cur_pts);
    reverse_pts = cur_pts;
    for (unsigned int i = 0; i < prev_pts.size(); i++) {
      auto &pt = reverse_pts[i];
      if (type == LEFT_RIGHT_IMG_MATCH && status[i] == 1) {
        pt.x -= move_cols;
      }
      if (type == RIGHT_LEFT_IMG_MATCH && status[i] == 1) {
        pt.x += move_cols;
      }
    }
    cv::cuda::GpuMat reverse_gpu_pts(reverse_pts);
    d_pyrLK_sparse->calc(gpu_cur_img, gpu_prev_img, gpu_cur_pts,
                         reverse_gpu_pts, reverse_gpu_status);
    reverse_gpu_pts.download(reverse_pts);
    reverse_gpu_status.download(reverse_status);
#else
    SPDLOG_ERROR("CUDA is not enabled");
#endif
  } else {
    cv::calcOpticalFlowPyrLK(
        prev_img, cur_img, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
    reverse_pts = cur_pts;
    for (unsigned int i = 0; i < prev_pts.size(); i++) {
      auto &pt = reverse_pts[i];
      if (type == LEFT_RIGHT_IMG_MATCH && status[i] == 1) {
        pt.x -= move_cols;
      }
      if (type == RIGHT_LEFT_IMG_MATCH && status[i] == 1) {
        pt.x += move_cols;
      }
    }
    cv::calcOpticalFlowPyrLK(
        cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE,
        PYR_LEVEL,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                         0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);
  }

  for (size_t i = 0; i < status.size(); i++) {
    if (status[i] && reverse_status[i] &&
        cv::norm(prev_pts[i] - reverse_pts[i]) <= 0.5) {
      status[i] = 1;
    } else
      status[i] = 0;
  }

  for (unsigned int i = 0; i < cur_pts.size(); i++) {
    if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
      status[i] = 0;
    }
  }
  reduceVector(prev_pts, status);
  reduceVector(cur_pts, status);
  reduceVector(ids, status);
  return cur_pts;
}

#ifdef USE_CUDA
LKImageInfoGPU opticalflowTrackPyr(const cv::Mat &cur_img,
                                   const LKImageInfoGPU &prev_lk,
                                   TrackLRType type) {
  cv::cuda::GpuMat gpu_cur_img(cur_img);
  auto cur_pyr = buildImagePyramid(gpu_cur_img, PYR_LEVEL);
  auto ids = prev_lk.lk_ids;
  auto prev_pts = prev_lk.lk_pts;
  auto prev_types = prev_lk.lk_types;
  std::vector<int> prev_local_index = prev_lk.lk_local_index;
  std::vector<Eigen::Vector3d> lk_pts_3d_norm;
  if (prev_pts.size() == 0) {
    LKImageInfoGPU ret;
    ret.pyr = cur_pyr;
    return ret;
  }
  TicToc tic;
  std::vector<uchar> status;
  std::vector<cv::Point2f> cur_pts;
  float move_cols =
      cur_img.cols * 90.0 /
      params->undistort_fov;  // slightly lower than 0.5 cols when fov=200

  if (type == WHOLE_IMG_MATCH) {
    cur_pts = prev_pts;
  } else {
    status.resize(prev_pts.size());
    std::fill(status.begin(), status.end(), 0);
    if (type == LEFT_RIGHT_IMG_MATCH) {
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
    reduceVector(prev_types, status);
    reduceVector(prev_local_index, status);
    reduceVector(ids, status);
  }
  status.resize(0);
  if (cur_pts.size() == 0) {
    LKImageInfoGPU ret;
    ret.pyr = cur_pyr;
    return ret;
  }
  std::vector<float> err;
  std::vector<uchar> reverse_status;
  std::vector<cv::Point2f> reverse_pts;

  cv::cuda::GpuMat gpu_prev_pts(prev_pts);
  cv::cuda::GpuMat gpu_cur_pts(cur_pts);
  cv::cuda::GpuMat gpu_status;
  cv::cuda::GpuMat reverse_gpu_status;
  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse =
      cv::cuda::SparsePyrLKOpticalFlow::create(WIN_SIZE, PYR_LEVEL, 30, true);
  d_pyrLK_sparse->calc(prev_lk.pyr, cur_pyr, gpu_prev_pts, gpu_cur_pts,
                       gpu_status);
  gpu_status.download(status);
  gpu_cur_pts.download(cur_pts);
  reverse_pts = cur_pts;
  for (unsigned int i = 0; i < prev_pts.size(); i++) {
    auto &pt = reverse_pts[i];
    if (type == LEFT_RIGHT_IMG_MATCH && status[i] == 1) {
      pt.x -= move_cols;
    }
    if (type == RIGHT_LEFT_IMG_MATCH && status[i] == 1) {
      pt.x += move_cols;
    }
  }
  cv::cuda::GpuMat reverse_gpu_pts(reverse_pts);
  d_pyrLK_sparse->calc(cur_pyr, prev_lk.pyr, gpu_cur_pts, reverse_gpu_pts,
                       reverse_gpu_status);
  reverse_gpu_pts.download(reverse_pts);
  reverse_gpu_status.download(reverse_status);

  for (size_t i = 0; i < status.size(); i++) {
    if (status[i] && reverse_status[i] &&
        cv::norm(prev_pts[i] - reverse_pts[i]) <= 0.5) {
      status[i] = 1;
    } else
      status[i] = 0;
  }

  for (unsigned int i = 0; i < cur_pts.size(); i++) {
    if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
      status[i] = 0;
    }
  }
  reduceVector(cur_pts, status);
  reduceVector(ids, status);
  reduceVector(prev_types, status);
  reduceVector(prev_local_index, status);
  return {cur_pts, lk_pts_3d_norm, ids, prev_local_index, prev_types, cur_pyr};
}
#endif

LKImageInfoCPU opticalflowTrackPyr(const cv::Mat &cur_img,
                                   const LKImageInfoCPU &prev_lk,
                                   TrackLRType type) {
  auto cur_pyr = buildImagePyramid(cur_img, PYR_LEVEL);
  auto ids = prev_lk.lk_ids;
  auto prev_pts = prev_lk.lk_pts;
  auto prev_types = prev_lk.lk_types;
  std::vector<int> prev_local_index = prev_lk.lk_local_index;
  std::vector<Eigen::Vector3d> lk_pts_3d_norm;
  if (prev_pts.size() == 0) {
    LKImageInfoCPU ret;
    ret.pyr = cur_pyr;
    return ret;
  }
  TicToc tic;
  std::vector<uchar> status;
  std::vector<cv::Point2f> cur_pts;
  float move_cols =
      cur_img.cols * 90.0 /
      params->undistort_fov;  // slightly lower than 0.5 cols when fov=200

  if (type == WHOLE_IMG_MATCH) {
    cur_pts = prev_pts;
  } else {
    status.resize(prev_pts.size());
    std::fill(status.begin(), status.end(), 0);
    if (type == LEFT_RIGHT_IMG_MATCH) {
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
    reduceVector(prev_types, status);
    reduceVector(prev_local_index, status);
    reduceVector(ids, status);
  }
  status.resize(0);
  if (cur_pts.size() == 0) {
    LKImageInfoCPU ret;
    ret.pyr = cur_pyr;
    return ret;
  }
  std::vector<float> err;
  std::vector<uchar> reverse_status;
  std::vector<cv::Point2f> reverse_pts;
  //TODO: Use pyr correctly
  cv::calcOpticalFlowPyrLK(prev_lk.pyr[0], cur_pyr[0], prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL);
  reverse_pts = cur_pts;
  for (unsigned int i = 0; i < prev_pts.size(); i++) {
    auto &pt = reverse_pts[i];
    if (type == LEFT_RIGHT_IMG_MATCH && status[i] == 1) {
      pt.x -= move_cols;
    }
    if (type == RIGHT_LEFT_IMG_MATCH && status[i] == 1) {
      pt.x += move_cols;
    }
  }
  //TODO: Use pyr correctly
  cv::calcOpticalFlowPyrLK(cur_pyr[0], prev_lk.pyr[0], cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL);

  for (size_t i = 0; i < status.size(); i++) {
    if (status[i] && reverse_status[i] &&
        cv::norm(prev_pts[i] - reverse_pts[i]) <= 0.5) {
      status[i] = 1;
    } else
      status[i] = 0;
  }

  for (unsigned int i = 0; i < cur_pts.size(); i++) {
    if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
      status[i] = 0;
    }
  }
  reduceVector(cur_pts, status);
  reduceVector(ids, status);
  reduceVector(prev_types, status);
  reduceVector(prev_local_index, status);
  return {cur_pts, lk_pts_3d_norm, ids, prev_local_index, prev_types, cur_pyr};
}

void detectPoints(const cv::Mat &img, std::vector<cv::Point2f> &n_pts,
                  const std::vector<cv::Point2f> &cur_pts, int require_pts,
                  bool enable_cuda, bool use_fast, int fast_rows,
                  int fast_cols) {
  int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());
  cv::Mat mask;
  if (params->enable_perf_output) {
    ROS_INFO("Lost %d pts; Require %d will detect %d", lack_up_top_pts,
             require_pts, lack_up_top_pts > require_pts / 4);
  }
  std::vector<cv::Point2f> n_pts_tmp;
  if (lack_up_top_pts > require_pts / 4) {
    int num_to_detect = lack_up_top_pts;
    if (cur_pts.size() > 0) {
      // We have some points, so try to detect slightly more points to
      // avoid overlap with current points
      num_to_detect = lack_up_top_pts * 2;
    }
    cv::Mat d_prevPts;
    if (use_fast) {
      n_pts_tmp =
          detectFastByRegion(img, mask, num_to_detect, fast_rows, fast_cols);
    } else {
      // Use goodFeaturesToTrack
      if (enable_cuda) {
#ifdef USE_CUDA
        cv::Ptr<cv::cuda::CornersDetector> detector =
            cv::cuda::createGoodFeaturesToTrackDetector(
                img.type(), num_to_detect, 0.01, params->feature_min_dist);
        cv::cuda::GpuMat d_prevPts_gpu;
        cv::cuda::GpuMat img_cuda(img);
        detector->detect(img_cuda, d_prevPts_gpu);
        d_prevPts_gpu.download(d_prevPts);
#else
        SPDLOG_ERROR("CUDA is not enabled");
#endif
      } else {
        cv::goodFeaturesToTrack(img, d_prevPts, num_to_detect, 0.01,
                                params->feature_min_dist, mask);
      }
      if (!d_prevPts.empty()) {
        n_pts_tmp = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
      } else {
        n_pts_tmp.clear();
      }
    }
    n_pts.clear();
    std::vector<cv::Point2f> all_pts = cur_pts;
    for (auto &pt : n_pts_tmp) {
      bool has_nearby = false;
      for (auto &pt_j : all_pts) {
        if (cv::norm(pt - pt_j) < params->feature_min_dist) {
          has_nearby = true;
          break;
        }
      }
      if (!has_nearby) {
        n_pts.push_back(pt);
        all_pts.push_back(pt);
      }
      if (n_pts.size() >= lack_up_top_pts) {
        break;
      }
    }
  } else {
    n_pts.clear();
  }
}

std::vector<cv::Point2f> detectFastByRegion(cv::InputArray _img,
                                            cv::InputArray _mask, int features,
                                            int cols, int rows) {
  int small_width = _img.cols() / cols;
  int small_height = _img.rows() / rows;
  int num_features = ceil((double)features * 1.5 / ((double)cols * rows));
#ifdef USE_CUDA
  auto fast = cv::cuda::FastFeatureDetector::create(
      10, true, cv::FastFeatureDetector::TYPE_9_16, features);
  cv::cuda::GpuMat gpu_img(_img);
#else
    auto fast = cv::FastFeatureDetector::create(10, true,
                                                cv::FastFeatureDetector::TYPE_9_16);
    cv::Mat gpu_img = _img.getMat();
#endif
  std::vector<cv::KeyPoint> total_kpts;
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {
      std::vector<cv::KeyPoint> kpts;
      cv::Rect roi(small_width * i, small_height * j, small_width,
                   small_height);
      fast->detect(gpu_img(roi), kpts);
      // printf("Detect %d features in region %d %d\n", kpts.size(), i,
      // j);
      for (auto kp : kpts) {
        kp.pt.x = kp.pt.x + small_width * i;
        kp.pt.y = kp.pt.y + small_height * j;
        total_kpts.push_back(kp);
      }
    }
  }
  // Sort the keypoints by confidence
  std::vector<cv::Point2f> ret;
  if (total_kpts.size() == 0) {
    return ret;
  }
  std::sort(total_kpts.begin(), total_kpts.end(),
            [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
              return a.response > b.response;
            });
  // Return the top features
  for (unsigned int i = 0; i < total_kpts.size(); i++) {
    ret.push_back(total_kpts[i].pt);
    if (ret.size() >= features) {
      break;
    }
  }
  return ret;
}

bool checkPointsWithinBounds(const std::vector<cv::Point2f>& pts, const cv::Size& img_size, const cv::Size& win_size, std::vector<cv::Point2f>& filtered_pts) {
    bool all_points_within_bounds = true;
    filtered_pts.clear();
    for (size_t i = 0; i < pts.size(); ++i) {
        cv::Point2f pt = pts[i];
        if (pt.x >= win_size.width && pt.y >= win_size.height &&
            pt.x + win_size.width < img_size.width &&
            pt.y + win_size.height < img_size.height) {
            filtered_pts.push_back(pt);
        }
    }
    return all_points_within_bounds;
}

std::vector<cv::Mat> buildImagePyramid(const cv::Mat &prevImg,
                                                int maxLevel_) {
  std::vector<cv::Mat> prevPyr;
  prevPyr.resize(maxLevel_ + 1);

  int cn = prevImg.channels();

  CV_Assert(cn == 1 || cn == 3 || cn == 4);

  prevPyr[0] = prevImg;
  for (int level = 1; level <= maxLevel_; ++level) {
    cv::pyrDown(prevPyr[level - 1], prevPyr[level]);
  }

  return prevPyr;
}

#ifdef USE_CUDA
std::vector<cv::cuda::GpuMat> buildImagePyramid(const cv::cuda::GpuMat &prevImg,
                                                int maxLevel_) {
  std::vector<cv::cuda::GpuMat> prevPyr;
  prevPyr.resize(maxLevel_ + 1);

  int cn = prevImg.channels();

  CV_Assert(cn == 1 || cn == 3 || cn == 4);

  prevPyr[0] = prevImg;
  for (int level = 1; level <= maxLevel_; ++level) {
    cv::cuda::pyrDown(prevPyr[level - 1], prevPyr[level]);
  }

  return prevPyr;
}
#endif
}  // namespace D2FrontEnd