#pragma once

#include <vector>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace Eigen;

namespace utils {
class MotionEstimator {
  public:
    bool
    solveRelativeRT(const std::vector<std::pair<Vector3d, Vector3d>> &corres,
                    Matrix3d &R, Vector3d &T);

  private:
    double testTriangulation(const std::vector<cv::Point2f> &l,
                             const std::vector<cv::Point2f> &r, cv::Mat_<double> R,
                             cv::Mat_<double> t);
    void decomposeE(cv::Mat E, cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};
} // namespace utils
