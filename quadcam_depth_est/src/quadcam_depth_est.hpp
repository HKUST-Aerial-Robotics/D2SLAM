#pragma once
#include <swarm_msgs/Pose.h>
#include <opencv2/cudaimgproc.hpp>
#include "virtual_stereo.hpp"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud.h>

namespace D2QuadCamDepthEst {

class QuadCamDepthEst {
    std::vector<Swarm::Pose> raw_cam_extrinsics;
    std::vector<Swarm::Pose> virtual_left_extrinsics;
    std::vector<VirtualStereo*> virtual_stereos;
    std::vector<D2Common::FisheyeUndist*> undistortors;
    HitnetONNX * hitnet = nullptr;
    int width = 320, height = 240;
    int pixel_step = 1;
    int image_step = 1;
    bool enable_texture = false;
    bool show;
    int image_count = 0;
    double min_z = 0.1;
    double max_z = 10;

    ros::NodeHandle nh;
    image_transport::ImageTransport * it_;
    image_transport::Subscriber image_sub;
    ros::Publisher pub_pcl;
    sensor_msgs::PointCloud pcl;
    void loadCNN(YAML::Node & config);
    void loadCameraConfig(YAML::Node & config, std::string configPath);
    void imageCallback(const sensor_msgs::ImageConstPtr & left);
public:
    QuadCamDepthEst(ros::NodeHandle & _nh);
    void inputImages(std::vector<cv::Mat> input_imgs);
};
}