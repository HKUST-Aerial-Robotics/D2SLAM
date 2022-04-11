#pragma once
#include "ros/ros.h"
#include <iostream>
#include "swarm_loop/loop_net.h"
#include "swarm_loop/loop_cam.h"
#include "swarm_loop/loop_detector.h"
#include <chrono> 
#include <Eigen/Eigen>
#include <thread>
#include <nav_msgs/Odometry.h>
#include <mutex>
#include <swarm_msgs/FisheyeFrameDescriptor.h>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/CompressedImage.h>

using namespace std::chrono; 

namespace swarm_localization_pkg {
class SwarmLoop {
protected:
    LoopDetector * loop_detector = nullptr;
    LoopCam * loop_cam = nullptr;
    LoopNet * loop_net = nullptr;
    ros::Subscriber cam_sub;
    bool debug_image = false;
    double min_movement_keyframe = 0.3;
    int self_id = 0;
    bool received_image = false;
    ros::Time last_kftime;
    Eigen::Vector3d last_keyframe_position = Eigen::Vector3d(10000, 10000, 10000);

    std::set<ros::Time> received_keyframe_stamps;

    CameraConfig camera_configuration;

    void on_loop_connection (LoopEdge & loop_con, bool is_local = false);

    std::queue<StereoFrame> raw_stereo_images;
    std::mutex raw_stereo_image_lock;

    StereoFrame find_images_raw(const nav_msgs::Odometry & odometry);

    void flatten_raw_callback(const vins::FlattenImages & viokf);

    void stereo_images_callback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    void comp_stereo_images_callback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::CompressedImageConstPtr right);
    void comp_depth_images_callback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    void depth_images_callback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr depth);
    double last_invoke = 0;
    
    void odometry_callback(const nav_msgs::Odometry & odometry);

    void odometry_keyframe_callback(const nav_msgs::Odometry & odometry);

    void VIOnonKF_callback(const StereoFrame & viokf);
    void VIOKF_callback(const StereoFrame & viokf, bool nonkeyframe = false);

    void pub_node_frame(const FisheyeFrameDescriptor_t & viokf);

    void on_remote_frame_ros(const swarm_msgs::FisheyeFrameDescriptor & remote_img_desc);

    void on_remote_image(const FisheyeFrameDescriptor_t & frame_desc);

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


    bool enable_pub_remote_frame;
    bool enable_pub_local_frame;
    bool enable_sub_remote_frame;
    bool send_img;
    bool send_whole_img_desc;
    std::thread th;

    double max_freq = 1.0;
    double recv_msg_duration = 0.5;
    double superpoint_thres = 0.012;
    int superpoint_max_num = 200;

    ros::Timer timer;

    geometry_msgs::Pose left_extrinsic, right_extrinsic;
public:
    SwarmLoop ();
    
protected:
    virtual void Init(ros::NodeHandle & nh);
};

}