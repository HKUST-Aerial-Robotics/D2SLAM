#pragma once

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <chrono> 

cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::Image &img_msg);
cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg, int flag);
inline int generate_keyframe_id(ros::Time stamp, int self_id) {
    static int keyframe_count = 0;
    int t_ms = 0;//stamp.toSec()*1000;
    return (t_ms%100000)*10000 + self_id*1000000 + keyframe_count++;
}


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
