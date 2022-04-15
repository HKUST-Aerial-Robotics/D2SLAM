#pragma once
#include "ros/ros.h"
#include <iostream>
#include "d2frontend/loop_net.h"
#include "d2frontend/loop_cam.h"
#include "d2frontend/loop_detector.h"
#include "d2frontend/d2featuretracker.h"
#include <chrono> 
#include <Eigen/Eigen>
#include <thread>
#include <nav_msgs/Odometry.h>
#include <mutex>
#include <swarm_msgs/FisheyeFrameDescriptor.h>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/CompressedImage.h>

using namespace std::chrono; 

namespace D2Frontend {
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

    void on_loop_connection (LoopEdge & loop_con, bool is_local = false);

    std::queue<StereoFrame> raw_stereo_images;
    std::mutex raw_stereo_image_lock;

    StereoFrame find_images_raw(const nav_msgs::Odometry & odometry);

    // void flatten_raw_callback(const ::FlattenImages & viokf);
    void stereo_images_callback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    void comp_stereo_images_callback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::CompressedImageConstPtr right);
    void comp_depth_images_callback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    void depth_images_callback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr depth);
    double last_invoke = 0;
    
    void odometry_callback(const nav_msgs::Odometry & odometry);

    void odometry_keyframe_callback(const nav_msgs::Odometry & odometry);

    void VIOnonKF_callback(const StereoFrame & viokf);
    void VIOKF_callback(const StereoFrame & viokf, bool nonkeyframe = false);

    void pub_node_frame(const VisualImageDescArray & viokf);

    void on_remote_frame_ros(const swarm_msgs::FisheyeFrameDescriptor & remote_img_desc);

    void on_remote_image(const VisualImageDescArray & frame_desc);

    void process_stereoframe(const StereoFrame & stereoframe);

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