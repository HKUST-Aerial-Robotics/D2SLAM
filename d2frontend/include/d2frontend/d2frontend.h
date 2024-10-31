#pragma once
#include "ros/ros.h"
#include <iostream>
#include <chrono> 
#include <Eigen/Eigen>
#include <thread>
#include <nav_msgs/Odometry.h>
#include <mutex>
#include <swarm_msgs/ImageArrayDescriptor.h>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include "d2common/d2frontend_types.h"
#include "d2frontend_params.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <queue>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std::chrono; 
using namespace swarm_msgs;
using namespace D2Common;

namespace D2FrontEnd {
class LoopCam;
class LoopNet;
class D2FeatureTracker;
class LoopDetector;
class D2Frontend {
    typedef image_transport::SubscriberFilter ImageSubscriber;

protected:
    using ApproSync = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;

    LoopDetector * loop_detector = nullptr;
    LoopCam * loop_cam = nullptr;
    LoopNet * loop_net = nullptr;
    D2FeatureTracker * feature_tracker = nullptr;
    ros::Subscriber cam_sub;
    ros::Time last_kftime;
    Eigen::Vector3d last_keyframe_position = Eigen::Vector3d(10000, 10000, 10000);

    std::set<ros::Time> received_keyframe_stamps;
    std::queue<VisualImageDescArray> loop_queue;
    std::mutex loop_lock;
    image_transport::ImageTransport * it_;
    virtual void Init(ros::NodeHandle & nh);
    virtual void backendFrameCallback(const VisualImageDescArray & viokf) {};

    void onLoopConnection (LoopEdge & loop_con, bool is_local = false);

    std::queue<StereoFrame> raw_stereo_images;
    std::mutex raw_stereo_image_lock;

    StereoFrame findImagesRaw(const nav_msgs::Odometry & odometry);

    // void flatten_raw_callback(const ::FlattenImages & viokf);
    void stereoImagesCallback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    void depthImagesCallback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr depth);
    void monoImageCallback(const sensor_msgs::ImageConstPtr & left);
    double last_invoke = 0;
    
    void pubNodeFrame(const VisualImageDescArray & viokf);

    void onRemoteFrameROS(const swarm_msgs::ImageArrayDescriptor & remote_img_desc);

    void onRemoteImage(VisualImageDescArray frame_desc);
    virtual void processRemoteImage(VisualImageDescArray & frame_desc, bool succ_track);

    void processStereoframe(const StereoFrame & stereoframe);

    void addToLoopQueue(const VisualImageDescArray & viokf);

    ros::Subscriber remote_img_sub;
    ros::Publisher loopconn_pub;
    ros::Publisher remote_image_desc_pub;
    ros::Publisher local_image_desc_pub;
    ros::Publisher keyframe_pub;

    ImageSubscriber * image_sub_l, *image_sub_r;
    message_filters::Synchronizer<ApproSync> * sync;
    image_transport::Subscriber image_sub_single;

    std::thread th, th_loop_det;
    bool received_image = false;
    ros::Timer timer, loop_timer;

    //data buffe
    std::queue<std::shared_ptr<StereoFrame>> stereo_frame_q_;

    std::mutex stereo_frame_buffer_lock_;
    int32_t visual_frame_size_ = 2;

    //process stereo frame thread
    std::thread stereo_frame_thread_;
    bool stereo_frame_thread_running_ = false;
    std::unique_ptr<ros::Rate> stereo_frame_thread_rate_ptr_;
    void processStereoFrameThread();
    
    //Loop detection thread
    std::thread loop_detection_thread_;
    bool loop_detection_thread_running_ = false;
    std::unique_ptr<ros::Rate> loop_detection_thread_rate_ptr_;
    void loopDetectionThread();

    //lcm thread
    std::thread lcm_thread_;
    bool lcm_thread_running_ = false;
    std::unique_ptr<ros::Rate> lcm_thread_rate_ptr_;
    void lcmThread();

    void startThread(){
        stereo_frame_thread_running_ = true;
        stereo_frame_thread_ = std::thread(&D2Frontend::processStereoFrameThread, this);
        loop_detection_thread_running_ = true;
        if (params->enable_loop)
        {
            loop_detection_thread_ = std::thread(&D2Frontend::loopDetectionThread, this);
            lcm_thread_running_ = true;
            lcm_thread_ = std::thread(&D2Frontend::lcmThread, this);
        }
    }

    void stopThreads(){
        stereo_frame_thread_running_ = false;
        stereo_frame_thread_.join();
        loop_detection_thread_running_ = false;
        loop_detection_thread_.join();
        lcm_thread_running_ = false;
        lcm_thread_.join();
    };



public:
    D2Frontend ();
    virtual Swarm::Pose getMotionPredict(double stamp) const {return Swarm::Pose();};
    void stopFrontend(){
        stopThreads();
    };
    
};

}