#pragma once
#include <opencv2/opencv.hpp>

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <chrono> 

namespace D2Frontend {
cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::Image &img_msg);
cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg, int flag);

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

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
}