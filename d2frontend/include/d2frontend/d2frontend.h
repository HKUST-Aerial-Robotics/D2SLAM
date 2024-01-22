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
    virtual void backendFrameCallback(const VisualImageDescArray & viokf) {};
    void onLoopConnection (LoopEdge & loop_con, bool is_local = false);
    StereoFrame findImagesRaw(const nav_msgs::Odometry & odometry);
    // void flatten_raw_callback(const ::FlattenImages & viokf);
    void stereoImagesCallback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right);
    void depthImagesCallback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr depth);
    void monoImageCallback(const sensor_msgs::ImageConstPtr & left);
    void pubNodeFrame(const VisualImageDescArray & viokf);
    void onRemoteFrameROS(const swarm_msgs::ImageArrayDescriptor & remote_img_desc);
    void onRemoteImage(VisualImageDescArray frame_desc);
    virtual void processRemoteImage(VisualImageDescArray & frame_desc, bool succ_track);
    void processStereoThread(int32_t fps);
    void processStereoframe(const StereoFrame & stereoframe);
    void loopDetectionThread();
    void addToLoopQueue(const VisualImageDescArray & viokf);

    std::queue<StereoFrame> raw_stereo_images;
    std::mutex raw_stereo_image_lock;
    double last_invoke = 0;   
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

    std::thread frontend_thread_;
    int32_t frontend_thread_running_ = 1;
    std::unique_ptr<ros::Rate> frontend_rate_ = nullptr;
    
    std::mutex stereo_frame_lock_;
    std::list<StereoFrame> stereo_frames_;
    std::shared_ptr<StereoFrame> current_stereo_frame_ = nullptr;
    int32_t stereo_frame_count_ = 0;

    int32_t loop_closure_detecting_running_ = 1;
    std::unique_ptr<ros::Rate> loop_closure_detecting_rate_ = nullptr;
    
    const int32_t kMaxQueueSize = 10;

public:
    D2Frontend(){};
    ~D2Frontend(){
        frontend_rate_ = nullptr;
        loop_closure_detecting_rate_ = nullptr;
    };
    void stopFrontend();
    virtual Swarm::Pose getMotionPredict(double stamp) const {return Swarm::Pose();};
protected:
    virtual void Init(ros::NodeHandle & nh);
};

}