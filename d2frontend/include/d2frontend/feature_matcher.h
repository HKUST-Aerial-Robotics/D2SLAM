#pragma once
#include <opencv2/opencv.hpp>

namespace D2FrontEnd {

std::vector<cv::DMatch>
matchKNN(const cv::Mat &desc_a, const cv::Mat &desc_b,
         double knn_match_ratio = 0.8,
         const std::vector<cv::Point2f> pts_a = std::vector<cv::Point2f>(),
         const std::vector<cv::Point2f> pts_b = std::vector<cv::Point2f>(),
         double search_local_dist = -1);

} // namespace D2FrontEnd