#pragma once
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <d2common/d2basetypes.h>
#include <d2frontend/d2frontend_params.h>
#include <d2common/d2landmarks.h>

namespace D2FrontEnd {
using D2Common::LandmarkIdType;
using LandmarkType = D2Common::LandmarkType;

cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::Image &img_msg);
cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg,
                        int flag);
Eigen::MatrixXf load_csv_mat_eigen(std::string csv);
Eigen::VectorXf load_csv_vec_eigen(std::string csv);

template <typename T, typename B>
inline void reduceVector(std::vector<T> &v, std::vector<B> status) {
    int j = 0;
    for (unsigned int i = 0; i < v.size(); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


cv::Vec3b extractColor(const cv::Mat &img, cv::Point2f p);
cv::Point2f rotate_pt_norm2d(cv::Point2f pt, Eigen::Quaterniond q);

} // namespace D2FrontEnd