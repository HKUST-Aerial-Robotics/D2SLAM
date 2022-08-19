#pragma once
#include <opencv2/opencv.hpp>

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <chrono> 

namespace D2FrontEnd {
cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::Image &img_msg);
cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg, int flag);
Eigen::MatrixXf load_csv_mat_eigen(std::string csv);
Eigen::VectorXf load_csv_vec_eigen(std::string csv);

template<typename T, typename B>
inline void reduceVector(std::vector<T> &v, std::vector<B> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


Swarm::Pose AffineRestoCamPose(Eigen::Matrix4d affine);

void PnPInitialFromCamPose(const Swarm::Pose &p, cv::Mat & rvec, cv::Mat & tvec);
Swarm::Pose PnPRestoCamPose(cv::Mat rvec, cv::Mat tvec);
cv::Vec3b extractColor(const cv::Mat &img, cv::Point2f p);
cv::Point2f rotate_pt_norm2d(cv::Point2f pt, Eigen::Quaterniond q);
}