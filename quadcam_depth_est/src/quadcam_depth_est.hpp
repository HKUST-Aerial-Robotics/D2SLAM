#pragma once
#include <swarm_msgs/Pose.h>
#include <opencv2/cudaimgproc.hpp>
#include "../include/virtual_stereo.hpp"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <image_transport/image_transport.h>
#include "../include/pcl_utils.hpp"
#include <d2common/d2basetypes.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/time_synchronizer.h>

typedef image_transport::SubscriberFilter ImageSubscriber;

namespace D2QuadCamDepthEst {
using D2Common::CameraConfig;
class QuadCamDepthEst {
    std::vector<Swarm::Pose> raw_cam_extrinsics;
    std::vector<Swarm::Pose> virtual_left_extrinsics;
    std::vector<VirtualStereo*> virtual_stereos;
    std::vector<D2Common::FisheyeUndist*> undistortors;
    std::vector<camodocal::CameraPtr> raw_cameras;
    HitnetONNX * hitnet = nullptr;
    CREStereoONNX * crestereo = nullptr;
    std::string cnn_type = "crestereo";
    int width = 320, height = 240;
    int pixel_step = 1;
    int image_step = 1;
    bool enable_texture = false;
    bool show;
    int image_count = 0;
    double min_z = 0.1;
    double max_z = 10;
    bool cnn_rgb = false;
    bool enable_cnn;

    ros::NodeHandle nh;
    image_transport::ImageTransport * it_;
    image_transport::Subscriber image_sub;
    ImageSubscriber * left_sub, *right_sub;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> * sync;
    ros::Publisher pub_pcl;
    PointCloud * pcl = nullptr;
    PointCloudRGB * pcl_color = nullptr;
    CameraConfig camera_config = D2Common::STEREO_PINHOLE;
    
    void loadCNN(YAML::Node & config);
    void loadCameraConfig(YAML::Node & config, std::string configPath);
    void imageCallback(const sensor_msgs::ImageConstPtr & left);
    void stereoImagesCallback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    std::pair<cv::Mat, cv::Mat> intrinsicsFromNode(const YAML::Node & node);
public:
    QuadCamDepthEst(ros::NodeHandle & _nh);
};
}