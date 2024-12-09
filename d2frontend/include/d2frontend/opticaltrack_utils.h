#pragma once
#include <opencv2/opencv.hpp>

#include <chrono>
#include <d2common/d2basetypes.h>
#include <d2common/d2landmarks.h>

#define PYR_LEVEL 2

namespace D2FrontEnd {
using D2Common::LandmarkIdType;
using LandmarkType = D2Common::LandmarkType;

template <typename T> struct LKImageInfo {
    std::vector<cv::Point2f> lk_pts;
    std::vector<Eigen::Vector3d> lk_pts_3d_norm;
    std::vector<LandmarkIdType> lk_ids;
    std::vector<int> lk_local_index;
    std::vector<LandmarkType> lk_types;
    std::vector<T> pyr;
};

using LKImageInfoCPU = LKImageInfo<cv::Mat>;
using LKImageInfoGPU = LKImageInfo<cv::cuda::GpuMat>;

void detectPoints(const cv::Mat &img, std::vector<cv::Point2f> &n_pts,
                  const std::vector<cv::Point2f> &cur_pts, int require_pts,
                  bool enable_cuda = true, bool use_fast = false,
                  int fast_rows = 3, int fast_cols = 4);

std::vector<cv::cuda::GpuMat> buildImagePyramid(const cv::cuda::GpuMat &prevImg,
                                                int maxLevel_ = 3);

std::vector<cv::Mat> buildImagePyramid(const cv::Mat &prevImg,
                                                int maxLevel_ = 3);

std::vector<cv::Point2f> opticalflowTrack(const cv::Mat &cur_img,
                                          const cv::Mat &prev_img,
                                          std::vector<cv::Point2f> &prev_pts,
                                          std::vector<LandmarkIdType> &ids,
                                          TrackLRType type = WHOLE_IMG_MATCH,
                                          bool enable_cuda = true);

LKImageInfoGPU opticalflowTrackPyr(
    const cv::Mat &cur_img, const LKImageInfoGPU& prev_lk,
    TrackLRType type = WHOLE_IMG_MATCH);

LKImageInfoCPU opticalflowTrackPyr(
    const cv::Mat &cur_img, const LKImageInfoCPU& prev_lk,
    TrackLRType type = WHOLE_IMG_MATCH);

std::vector<cv::DMatch>
matchKNN(const cv::Mat &desc_a, const cv::Mat &desc_b,
         double knn_match_ratio = 0.8,
         const std::vector<cv::Point2f> pts_a = std::vector<cv::Point2f>(),
         const std::vector<cv::Point2f> pts_b = std::vector<cv::Point2f>(),
         double search_local_dist = -1);

} // namespace D2FrontEnd