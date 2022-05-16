#pragma once
#include "ros/ros.h"
#include <iostream>
#include <chrono> 
#include <Eigen/Eigen>
#include <thread>
#include <nav_msgs/Odometry.h>
#include <mutex>
#include <swarm_msgs/ImageArrayDescriptor.h>
#include <swarm_msgs/swarm_types.hpp>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include "d2frontend_types.h"
#include "d2frontend_params.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

using namespace std::chrono; 
using namespace swarm_msgs;

namespace D2FrontEnd {
class LoopCam;
class LoopNet;
class D2FeatureTracker;
class LoopDetector;
class D2Frontend {
protected:
    LoopDetector * loop_detector = nullptr;
    LoopCam * loop_cam = nullptr;
    LoopNet * loop_net = nullptr;
    D2FeatureTracker * feature_tracker = nullptr;
    ros::Subscriber cam_sub;
    ros::Time last_kftime;
    Eigen::Vector3d last_keyframe_position = Eigen::Vector3d(10000, 10000, 10000);

    std::set<ros::Time> received_keyframe_stamps;

    virtual void frameCallback(const VisualImageDescArray & viokf) {};

    void onLoopConnection (LoopEdge & loop_con, bool is_local = false);

    std::queue<StereoFrame> raw_stereo_images;
    std::mutex raw_stereo_image_lock;

    StereoFrame findImagesRaw(const nav_msgs::Odometry & odometry);

    // void flatten_raw_callback(const ::FlattenImages & viokf);
    void stereoImagesCallback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    void compStereoImagesCallback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::CompressedImageConstPtr right);
    void compDepthImagesCallback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    void depthImagesCallback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr depth);
    double last_invoke = 0;
    
    void odometryCallback(const nav_msgs::Odometry & odometry);

    void odometryKeyframeCallback(const nav_msgs::Odometry & odometry);

    void viononKFCallback(const StereoFrame & viokf);
    void vioKFCallback(const StereoFrame & viokf, bool nonkeyframe = false);

    void pubNodeFrame(const VisualImageDescArray & viokf);

    void onRemoteFrameROS(const swarm_msgs::ImageArrayDescriptor & remote_img_desc);

    void onRemoteImage(const VisualImageDescArray & frame_desc);

    void processStereoframe(const StereoFrame & stereoframe);

    ros::Subscriber camera_sub;
    ros::Subscriber viokeyframe_sub;
    ros::Subscriber odometry_sub;
    ros::Subscriber keyframe_odometry_sub;
    ros::Subscriber flatten_raw_sub;
    ros::Subscriber remote_img_sub;
    ros::Subscriber viononkeyframe_sub;
    ros::Publisher loopconn_pub;
    ros::Publisher remote_image_desc_pub;
    ros::Publisher local_image_desc_pub;
    ros::Publisher keyframe_pub;

    message_filters::Subscriber<sensor_msgs::Image> * image_sub_l, *image_sub_r;
    message_filters::Subscriber<sensor_msgs::CompressedImage> * comp_image_sub_l, *comp_image_sub_r;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> * sync;
    message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> * comp_sync;
    message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::Image> * comp_depth_sync;


    std::thread th;
    bool received_image = false;
    ros::Timer timer;
public:
    D2Frontend ();
    
protected:
    virtual void Init(ros::NodeHandle & nh);
};

}