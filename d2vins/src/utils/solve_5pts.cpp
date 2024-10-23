#include "solve_5pts.h"

#include "spdlog/spdlog.h"

namespace utils {

bool MotionEstimator::solveRelativeRT(
    const std::vector<std::pair<Vector3d, Vector3d>> &corres,
    Matrix3d &Rotation, Vector3d &Translation) {
  if (corres.size() >= 15) {
    std::vector<cv::Point2f> ll, rr;
    for (unsigned int i = 0; i < corres.size(); i++) {
      if (fabs(corres[i].first(2)) > 0.001 &&
          fabs(corres[i].second(2)) > 0.001) {
        cv::Point2f first =
            cv::Point2f(corres[i].first(0) / corres[i].first(2),
                        corres[i].first(1) / corres[i].first(2));
        cv::Point2f second =
            cv::Point2f(corres[i].second(0) / corres[i].second(2),
                        corres[i].second(1) / corres[i].second(2));
        ll.push_back(first);
        rr.push_back(second);
      }
    }
    cv::Mat cameraMatrix =
        (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(ll, rr, cameraMatrix, cv::RANSAC, 0.99,
                                     0.3 / 460, 1000, mask);
    // cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460,
    // 0.99, mask);
    cv::Mat rot, trans;
    int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
    SPDLOG_INFO("solveRelativeRT inlier_cnt: {}/{}", inlier_cnt, corres.size());
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    for (int i = 0; i < 3; i++) {
      T(i) = trans.at<double>(i, 0);
      for (int j = 0; j < 3; j++) R(i, j) = rot.at<double>(i, j);
    }

    Rotation = R.transpose();
    Translation = -R.transpose() * T;
    if (inlier_cnt > 12)
      return true;
    else
      return false;
  }
  return false;
}

}  // namespace utils
