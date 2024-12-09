#include <d2frontend/feature_matcher.h>

namespace D2FrontEnd {
std::vector<cv::DMatch> matchKNN(const cv::Mat &desc_a, const cv::Mat &desc_b,
                                 double knn_match_ratio,
                                 const std::vector<cv::Point2f> pts_a,
                                 const std::vector<cv::Point2f> pts_b,
                                 double search_local_dist) {
  // Match descriptors with OpenCV knnMatch
  std::vector<std::vector<cv::DMatch>> matches;
  std::vector<std::vector<cv::DMatch>> matches_inv;
  cv::BFMatcher bfmatcher(cv::NORM_L2);
  bfmatcher.knnMatch(desc_a, desc_b, matches, 2);
  bfmatcher.knnMatch(desc_b, desc_a, matches_inv, 2);
  // Build up dict for matches_inv
  std::vector<int> match_inv_dict(desc_b.rows, -1);
  for (auto &match : matches_inv) {
    if (match.size() < 2) {
      continue;
    }
    if (match[0].distance < knn_match_ratio * match[1].distance) {
      match_inv_dict[match[0].queryIdx] = match[0].trainIdx;
    }
  }
  std::vector<cv::DMatch> good_matches;
  for (auto &match : matches) {
    if (match.size() < 2) {
      continue;
    }
    if (match[0].distance < knn_match_ratio * match[1].distance &&
        match_inv_dict[match[0].trainIdx] == match[0].queryIdx) {
      if (search_local_dist > 0) {
        if (cv::norm(pts_a[match[0].queryIdx] - pts_b[match[0].trainIdx]) >
            search_local_dist) {
          continue;
        }
      }
      good_matches.push_back(match[0]);
    }
  }
  return good_matches;
}
}
