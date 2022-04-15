#include <d2frontend/d2frontend.h>
#include <d2frontend/utils.h>

#include "ros/ros.h"
#include <iostream>
#include "d2frontend/loop_net.h"
#include "d2frontend/loop_cam.h"
#include "d2frontend/loop_detector.h"
#include <Eigen/Eigen>
#include <thread>
#include <nav_msgs/Odometry.h>
#include <mutex>
#include <swarm_msgs/node_frame.h>

#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}

namespace D2Frontend {
void D2Frontend::on_loop_connection (LoopEdge & loop_con, bool is_local) {
    if(is_local) {
        loop_net->broadcast_loop_connection(loop_con);
    }

    // ROS_INFO("Pub loop conn. is local %d", is_local);
    loopconn_pub.publish(loop_con);
}

StereoFrame D2Frontend::find_images_raw(const nav_msgs::Odometry & odometry) {
    // ROS_INFO("find_images_raw %f", odometry.header.stamp.toSec());
    auto stamp = odometry.header.stamp;
    StereoFrame ret;
    raw_stereo_image_lock.lock();
    while (raw_stereo_images.size() > 0 && stamp.toSec() - raw_stereo_images.front().stamp.toSec() > 1e-3) {
        // ROS_INFO("Removing d stamp %f", stamp.toSec() - raw_stereo_images.front().stamp.toSec());
        raw_stereo_images.pop();
    }

    if (raw_stereo_images.size() > 0 && fabs(stamp.toSec() - raw_stereo_images.front().stamp.toSec()) < 1e-3) {
        auto ret = raw_stereo_images.front();
        raw_stereo_images.pop();
        ret.pose_drone = odometry.pose.pose;
        // ROS_INFO("VIO KF found, returning...");
        raw_stereo_image_lock.unlock();
        return ret;
    } 

    raw_stereo_image_lock.unlock();
    return ret;
}

// void D2Frontend::flatten_raw_callback(const vins::FlattenImages & stereoframe) {
//     raw_stereo_image_lock.lock();
//     // ROS_INFO("(D2Frontend::flatten_raw_callback) Received flatten_raw %f", stereoframe.header.stamp.toSec());
//     raw_stereo_images.push(StereoFrame(stereoframe, params->self_id));
//     raw_stereo_image_lock.unlock();
// }

void D2Frontend::stereo_images_callback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right) {
    auto _l = getImageFromMsg(left);
    auto _r = getImageFromMsg(right);
    StereoFrame sframe(_l->header.stamp, _l->image, _r->image, params->left_extrinsic, params->right_extrinsic, params->self_id);
    process_stereoframe(sframe);
}

void D2Frontend::comp_stereo_images_callback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::CompressedImageConstPtr right) {
    auto _l = getImageFromMsg(left, cv::IMREAD_GRAYSCALE);
    auto _r = getImageFromMsg(right, cv::IMREAD_GRAYSCALE);
    StereoFrame sframe(left->header.stamp, _l, _r, params->left_extrinsic, params->right_extrinsic, params->self_id);
    process_stereoframe(sframe);
}

void D2Frontend::comp_depth_images_callback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::ImageConstPtr depth) {
    auto _l = getImageFromMsg(left, cv::IMREAD_GRAYSCALE);
    auto _d = getImageFromMsg(depth);
    StereoFrame sframe(left->header.stamp, _l, _d->image, params->left_extrinsic, params->self_id);
    process_stereoframe(sframe);
}

void D2Frontend::depth_images_callback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr depth) {
    auto _l = getImageFromMsg(left);
    auto _d = getImageFromMsg(depth);
    StereoFrame sframe(left->header.stamp, _l->image, _d->image, params->left_extrinsic, params->self_id);
    process_stereoframe(sframe);
}

void D2Frontend::odometry_callback(const nav_msgs::Odometry & odometry) {
    if (odometry.header.stamp.toSec() - last_invoke < params->ACCEPT_NONKEYFRAME_WAITSEC) {
        return;
    }

    auto _stereoframe = find_images_raw(odometry);
    if (_stereoframe.stamp.toSec() > 1000) {
        // ROS_INFO("VIO Non Keyframe callback!!");
        VIOnonKF_callback(_stereoframe);
    } else {
        // ROS_WARN("[D2Frontend] (odometry_callback) Flattened images correspond to this Odometry not found: %f", odometry.header.stamp.toSec());
    }
}

void D2Frontend::odometry_keyframe_callback(const nav_msgs::Odometry & odometry) {
    // ROS_INFO("VIO Keyframe received");
    auto _imagesraw = find_images_raw(odometry);
    if (_imagesraw.stamp.toSec() > 1000) {
        VIOKF_callback(_imagesraw);
    } else {
        // ROS_WARN("[SWARM_LOOP] (odometry_keyframe_callback) Flattened images correspond to this Keyframe not found: %f", odometry.header.stamp.toSec());
    }
}

void D2Frontend::VIOnonKF_callback(const StereoFrame & stereoframe) {
    if (!received_image && (stereoframe.stamp - last_kftime).toSec() > INIT_ACCEPT_NONKEYFRAME_WAITSEC) {
        // ROS_INFO("[SWARM_LOOP] (VIOnonKF_callback) USE non vio kf as KF at first keyframe!");
        VIOKF_callback(stereoframe);
        return;
    }
    
    //If never received image or 15 sec not receiving kf, use this as KF, this is ensure we don't missing data
    //Note that for the second case, we will not add it to database, matching only
    if ((stereoframe.stamp - last_kftime).toSec() > params->ACCEPT_NONKEYFRAME_WAITSEC) {
        VIOKF_callback(stereoframe, true);
    }
}

void D2Frontend::VIOKF_callback(const StereoFrame & stereoframe, bool nonkeyframe) {
}

void D2Frontend::process_stereoframe(const StereoFrame & stereoframe) {
    std::vector<cv::Mat> debug_imgs;
    static int count = 0;
    // ROS_INFO("[D2Frontend::process_stereoframe] %d", count ++);
    auto vframearry = loop_cam->process_stereoframe(stereoframe, debug_imgs);
    if (vframearry.landmark_num == 0) {
        ROS_WARN("[SWARM_LOOP] Null img desc, CNN no ready");
        return;
    }
    bool is_keyframe = feature_tracker->track(vframearry);
    vframearry.prevent_adding_db = !is_keyframe;
    vframearry.is_keyframe = is_keyframe;
    received_image = true;

    frame_callback(vframearry);

    if (is_keyframe) {
        //Do we need to wait for VIO?
        if (params->enable_network) {
            loop_net->broadcast_fisheye_desc(vframearry);
        }
        if (params->enable_loop) {
            loop_detector->on_image_recv(vframearry, debug_imgs);
        }
    }
}

void D2Frontend::pub_node_frame(const VisualImageDescArray & viokf) {
    ROS_INFO("[SWARM_LOOP](pub_node_frame) drone %d pub nodeframe", viokf.drone_id);
    swarm_msgs::node_frame nf;
    nf.header.stamp = ros::Time(viokf.stamp);
    nf.position.x = viokf.pose_drone.pos().x();
    nf.position.y = viokf.pose_drone.pos().y();
    nf.position.z = viokf.pose_drone.pos().z();
    nf.quat.x = viokf.pose_drone.att().x();
    nf.quat.y = viokf.pose_drone.att().y();
    nf.quat.z = viokf.pose_drone.att().z();
    nf.quat.w = viokf.pose_drone.att().w();
    nf.vo_available = true;
    nf.drone_id = viokf.drone_id;
    nf.keyframe_id = viokf.frame_id;
    keyframe_pub.publish(nf);
}

void D2Frontend::on_remote_frame_ros(const swarm_msgs::FisheyeFrameDescriptor & remote_img_desc) {
    // ROS_INFO("Remote");
    if (received_image) {
        this->on_remote_image(remote_img_desc);
    }
}

void D2Frontend::on_remote_image(const VisualImageDescArray & frame_desc) {
    ROS_INFO("Received from remote!");
    loop_detector->on_image_recv(frame_desc);
}

D2Frontend::D2Frontend () {}

void D2Frontend::Init(ros::NodeHandle & nh) {
    //Init Loop Net
    params = new D2FrontendParams(nh);
    cv::setNumThreads(1);

    loop_net = new LoopNet(params->_lcm_uri, params->send_img, params->send_whole_img_desc, params->recv_msg_duration);
    loop_cam = new LoopCam(*(params->loopcamconfig), nh);
    feature_tracker = new D2FeatureTracker(*(params->ftconfig));
        
    loop_cam->show = params->debug_image; 
    loop_detector = new LoopDetector(params->self_id, *(params->loopdetectorconfig));
    loop_detector->loop_cam = loop_cam;
    loop_detector->enable_visualize = params->debug_image;

    loop_detector->on_loop_cb = [&] (LoopEdge & loop_con) {
        this->on_loop_connection(loop_con, true);
    };

    loop_net->frame_desc_callback = [&] (const VisualImageDescArray & frame_desc) {
        if (received_image) {
            if (params->enable_pub_remote_frame) {
                remote_image_desc_pub.publish(frame_desc.toROS());
            }
            this->on_remote_image(frame_desc);
            this->pub_node_frame(frame_desc);
        }
    };

    loop_net->loopconn_callback = [&] (const LoopEdge_t & loop_conn) {
        auto loc = toROSLoopEdge(loop_conn);
        on_loop_connection(loc, false);
    };

    if (params->camera_configuration == CameraConfig::STEREO_FISHEYE) {
        // flatten_raw_sub = nh.subscribe("/vins_estimator/flattened_gray", 1, &D2Frontend::flatten_raw_callback, this, ros::TransportHints().tcpNoDelay());
    } else if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
        //Subscribe stereo pinhole, probrably is 
        if (params->is_comp_images) {
            ROS_INFO("[D2Frontend] Input: compressed images %s and %s", params->COMP_IMAGE0_TOPIC.c_str(), params->COMP_IMAGE1_TOPIC.c_str());
            comp_image_sub_l = new message_filters::Subscriber<sensor_msgs::CompressedImage> (nh, params->COMP_IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            comp_image_sub_r = new message_filters::Subscriber<sensor_msgs::CompressedImage> (nh, params->COMP_IMAGE1_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            comp_sync = new message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> (*comp_image_sub_l, *comp_image_sub_r, 1000);
            comp_sync->registerCallback(boost::bind(&D2Frontend::comp_stereo_images_callback, this, _1, _2));
        } else {
            ROS_INFO("[D2Frontend] Input: raw images %s and %s", params->IMAGE0_TOPIC.c_str(), params->IMAGE1_TOPIC.c_str());
            image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (nh, params->IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (nh, params->IMAGE1_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 1000);
            sync->registerCallback(boost::bind(&D2Frontend::stereo_images_callback, this, _1, _2));
        }
    } else if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
        if (params->is_comp_images) {
            ROS_INFO("[D2Frontend] Input: compressed images %s and depth %s", params->COMP_IMAGE0_TOPIC.c_str(), params->DEPTH_TOPIC.c_str());
            comp_image_sub_l = new message_filters::Subscriber<sensor_msgs::CompressedImage> (nh, params->COMP_IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (nh, params->DEPTH_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            comp_depth_sync = new message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::Image> (*comp_image_sub_l, *image_sub_r, 1000);
            comp_depth_sync->registerCallback(boost::bind(&D2Frontend::comp_depth_images_callback, this, _1, _2));
        } else {
            ROS_INFO("[D2Frontend] Input: raw images %s and depth %s", params->IMAGE0_TOPIC.c_str(), params->DEPTH_TOPIC.c_str());
            image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (nh, params->IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (nh, params->DEPTH_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 1000);
            sync->registerCallback(boost::bind(&D2Frontend::depth_images_callback, this, _1, _2));
        }
    }
    
    keyframe_pub = nh.advertise<swarm_msgs::node_frame>("keyframe", 10);
    odometry_sub  = nh.subscribe("/vins_estimator/odometry", 1, &D2Frontend::odometry_callback, this, ros::TransportHints().tcpNoDelay());
    keyframe_odometry_sub  = nh.subscribe("/vins_estimator/keyframe_pose", 1, &D2Frontend::odometry_keyframe_callback, this, ros::TransportHints().tcpNoDelay());

    loopconn_pub = nh.advertise<swarm_msgs::LoopEdge>("loop_connection", 10);
    
    if (params->enable_sub_remote_frame) {
        ROS_INFO("[SWARM_LOOP] Subscribing remote image from bag");
        remote_img_sub = nh.subscribe("/swarm_loop/remote_frame_desc", 1, &D2Frontend::on_remote_frame_ros, this, ros::TransportHints().tcpNoDelay());
    }

    if (params->enable_pub_remote_frame) {
        remote_image_desc_pub = nh.advertise<swarm_msgs::FisheyeFrameDescriptor>("remote_frame_desc", 10);
    }

    if (params->enable_pub_local_frame) {
        local_image_desc_pub = nh.advertise<swarm_msgs::FisheyeFrameDescriptor>("local_frame_desc", 10);
    }
    

    timer = nh.createTimer(ros::Duration(0.01), [&](const ros::TimerEvent & e) {
        loop_net->scan_recv_packets();
    });

    th = std::thread([&] {
        while(0 == loop_net->lcm_handle()) {
        }
    });
}

}
    