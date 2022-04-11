#include <swarm_loop/swarm_loop.h>
#include <swarm_loop/utils.h>

#include "ros/ros.h"
#include <iostream>
#include "swarm_loop/loop_net.h"
#include "swarm_loop/loop_cam.h"
#include "swarm_loop/loop_detector.h"
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
namespace swarm_localization_pkg {

void SwarmLoop::on_loop_connection (LoopEdge & loop_con, bool is_local) {
    if(is_local) {
        loop_net->broadcast_loop_connection(loop_con);
    }

    // ROS_INFO("Pub loop conn. is local %d", is_local);
    loopconn_pub.publish(loop_con);
}

StereoFrame SwarmLoop::find_images_raw(const nav_msgs::Odometry & odometry) {
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

void SwarmLoop::flatten_raw_callback(const vins::FlattenImages & stereoframe) {
    raw_stereo_image_lock.lock();
    // ROS_INFO("(SwarmLoop::flatten_raw_callback) Received flatten_raw %f", stereoframe.header.stamp.toSec());
    raw_stereo_images.push(StereoFrame(stereoframe, self_id));
    raw_stereo_image_lock.unlock();
}

void SwarmLoop::stereo_images_callback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right) {
    auto _l = getImageFromMsg(left);
    auto _r = getImageFromMsg(right);
    raw_stereo_image_lock.lock();
    raw_stereo_images.push(StereoFrame(_l->header.stamp, 
        _l->image, _r->image, left_extrinsic, right_extrinsic, self_id));
    raw_stereo_image_lock.unlock();
}


void SwarmLoop::comp_stereo_images_callback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::CompressedImageConstPtr right) {
    auto _l = getImageFromMsg(left, cv::IMREAD_GRAYSCALE);
    auto _r = getImageFromMsg(right, cv::IMREAD_GRAYSCALE);
    raw_stereo_image_lock.lock();
    raw_stereo_images.push(StereoFrame(left->header.stamp, 
        _l, _r, left_extrinsic, right_extrinsic, self_id));
    raw_stereo_image_lock.unlock();
}


void SwarmLoop::comp_depth_images_callback(const sensor_msgs::CompressedImageConstPtr left, const sensor_msgs::ImageConstPtr depth) {
    auto _l = getImageFromMsg(left, cv::IMREAD_GRAYSCALE);
    auto _d = getImageFromMsg(depth);
    raw_stereo_image_lock.lock();
    raw_stereo_images.push(StereoFrame(left->header.stamp, 
        _l, _d->image, left_extrinsic, self_id));
    raw_stereo_image_lock.unlock();
}

void SwarmLoop::depth_images_callback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr depth) {
    auto _l = getImageFromMsg(left);
    auto _d = getImageFromMsg(depth);
    raw_stereo_image_lock.lock();
    raw_stereo_images.push(StereoFrame(left->header.stamp, 
        _l->image, _d->image, left_extrinsic, self_id));
    raw_stereo_image_lock.unlock();
}

void SwarmLoop::odometry_callback(const nav_msgs::Odometry & odometry) {
    if (odometry.header.stamp.toSec() - last_invoke < ACCEPT_NONKEYFRAME_WAITSEC) {
        return;
    }

    auto _stereoframe = find_images_raw(odometry);
    if (_stereoframe.stamp.toSec() > 1000) {
        // ROS_INFO("VIO Non Keyframe callback!!");
        VIOnonKF_callback(_stereoframe);
    } else {
        // ROS_WARN("[SwarmLoop] (odometry_callback) Flattened images correspond to this Odometry not found: %f", odometry.header.stamp.toSec());
    }
}

void SwarmLoop::odometry_keyframe_callback(const nav_msgs::Odometry & odometry) {
    // ROS_INFO("VIO Keyframe received");
    auto _imagesraw = find_images_raw(odometry);
    if (_imagesraw.stamp.toSec() > 1000) {
        VIOKF_callback(_imagesraw);
    } else {
        ROS_WARN("[SWARM_LOOP] (odometry_keyframe_callback) Flattened images correspond to this Keyframe not found: %f", odometry.header.stamp.toSec());
    }
}

void SwarmLoop::VIOnonKF_callback(const StereoFrame & stereoframe) {
    if (!received_image && (stereoframe.stamp - last_kftime).toSec() > INIT_ACCEPT_NONKEYFRAME_WAITSEC) {
        //
        ROS_INFO("[SWARM_LOOP] (VIOnonKF_callback) USE non vio kf as KF at first keyframe!");
        VIOKF_callback(stereoframe);
        return;
    }
    
    //If never received image or 15 sec not receiving kf, use this as KF, this is ensure we don't missing data
    //Note that for the second case, we will not add it to database, matching only
    
    if ((stereoframe.stamp - last_kftime).toSec() > ACCEPT_NONKEYFRAME_WAITSEC) {
        VIOKF_callback(stereoframe, true);
    }
}

void SwarmLoop::VIOKF_callback(const StereoFrame & stereoframe, bool nonkeyframe) {
    Eigen::Vector3d drone_pos(stereoframe.pose_drone.position.x, stereoframe.pose_drone.position.y, stereoframe.pose_drone.position.z);
    double dpos = (last_keyframe_position - drone_pos).norm();

    if (stereoframe.stamp.toSec() - last_invoke < 1/max_freq) {
        return;
    }

    last_invoke = stereoframe.stamp.toSec();
    
    last_kftime = stereoframe.stamp;

    auto start = high_resolution_clock::now();
    std::vector<cv::Mat> imgs;
    
    auto ret = loop_cam->on_flattened_images(stereoframe, imgs);
    
    ret.prevent_adding_db = nonkeyframe && dpos < min_movement_keyframe;

    if (ret.landmark_num == 0) {
        ROS_WARN("[SWARM_LOOP] Null img desc, CNN no ready");
        return;
    }

    received_image = true;
    last_keyframe_position = drone_pos;

    loop_net->broadcast_fisheye_desc(ret);
    loop_detector->on_image_recv(ret, imgs);
    pub_node_frame(ret);
}

void SwarmLoop::pub_node_frame(const FisheyeFrameDescriptor_t & viokf) {
    ROS_INFO("[SWARM_LOOP](pub_node_frame) drone %d pub nodeframe", viokf.drone_id);
    swarm_msgs::node_frame nf;
    nf.header.stamp = toROSTime(viokf.timestamp);
    nf.position.x = viokf.pose_drone.position[0];
    nf.position.y = viokf.pose_drone.position[1];
    nf.position.z = viokf.pose_drone.position[2];
    nf.quat.x = viokf.pose_drone.orientation[0];
    nf.quat.y = viokf.pose_drone.orientation[1];
    nf.quat.z = viokf.pose_drone.orientation[2];
    nf.quat.w = viokf.pose_drone.orientation[3];
    nf.vo_available = true;
    nf.drone_id = viokf.drone_id;
    nf.keyframe_id = viokf.msg_id;
    keyframe_pub.publish(nf);
}


void SwarmLoop::on_remote_frame_ros(const swarm_msgs::FisheyeFrameDescriptor & remote_img_desc) {
    // ROS_INFO("Remote");
    if (received_image) {
        this->on_remote_image(toLCMFisheyeDescriptor(remote_img_desc));
    }
}

void SwarmLoop::on_remote_image(const FisheyeFrameDescriptor_t & frame_desc) {
    loop_detector->on_image_recv(frame_desc);
}


SwarmLoop::SwarmLoop () {}

void SwarmLoop::Init(ros::NodeHandle & nh) {
    //Init Loop Net
    std::string _lcm_uri = "0.0.0.0";
    std::string camera_config_path = "";
    std::string superpoint_model_path = "";
    std::string netvlad_model_path = "";
    std::string vins_config_path;
    std::string _pca_comp_path, _pca_mean_path;
    std::string IMAGE0_TOPIC, IMAGE1_TOPIC, COMP_IMAGE0_TOPIC, COMP_IMAGE1_TOPIC, DEPTH_TOPIC;
    cv::setNumThreads(1);
    nh.param<int>("self_id", self_id, -1);
    nh.param<bool>("is_4dof", is_4dof, true);
    nh.param<double>("min_movement_keyframe", min_movement_keyframe, 0.3);
    nh.param<double>("nonkeyframe_waitsec", ACCEPT_NONKEYFRAME_WAITSEC, 5.0);

    nh.param<std::string>("lcm_uri", _lcm_uri, "udpm://224.0.0.251:7667?ttl=1");
    
    nh.param<int>("init_loop_min_feature_num", INIT_MODE_MIN_LOOP_NUM, 10);
    nh.param<int>("match_index_dist", MATCH_INDEX_DIST, 10);
    nh.param<int>("min_loop_feature_num", MIN_LOOP_NUM, 15);
    nh.param<int>("min_match_per_dir", MIN_MATCH_PRE_DIR, 15);
    nh.param<int>("jpg_quality", JPG_QUALITY, 50);
    nh.param<int>("accept_min_3d_pts", ACCEPT_MIN_3D_PTS, 50);
    nh.param<int>("inter_drone_init_frames", inter_drone_init_frames, 50);
    nh.param<bool>("enable_lk", ENABLE_LK_LOOP_DETECTION, true);
    nh.param<bool>("enable_pub_remote_frame", enable_pub_remote_frame, false);
    nh.param<bool>("enable_pub_local_frame", enable_pub_local_frame, false);
    nh.param<bool>("enable_sub_remote_frame", enable_sub_remote_frame, false);
    nh.param<bool>("send_img", send_img, false);
    nh.param<bool>("is_pc_replay", IS_PC_REPLAY, false);
    nh.param<bool>("send_whole_img_desc", send_whole_img_desc, false);
    nh.param<bool>("send_all_features", SEND_ALL_FEATURES, false);
    nh.param<double>("query_thres", INNER_PRODUCT_THRES, 0.6);
    nh.param<double>("init_query_thres", INIT_MODE_PRODUCT_THRES, 0.3);
    nh.param<double>("max_freq", max_freq, 1.0);
    nh.param<double>("recv_msg_duration", recv_msg_duration, 0.5);
    nh.param<double>("superpoint_thres", superpoint_thres, 0.012);
    nh.param<int>("superpoint_max_num", superpoint_max_num, 200);
    nh.param<double>("detector_match_thres", DETECTOR_MATCH_THRES, 0.9);
    nh.param<bool>("lower_cam_as_main", LOWER_CAM_AS_MAIN, false);
    nh.param<bool>("output_raw_superpoint_desc", OUTPUT_RAW_SUPERPOINT_DESC, false);

    nh.param<double>("odometry_consistency_threshold", odometry_consistency_threshold, 2.0);
    nh.param<double>("pos_covariance_per_meter", pos_covariance_per_meter, 0.01);
    nh.param<double>("yaw_covariance_per_meter", yaw_covariance_per_meter, 0.003);

    nh.param<double>("triangle_thres", TRIANGLE_THRES, 0.006);
    nh.param<bool>("debug_no_rejection", DEBUG_NO_REJECT, false);
    nh.param<double>("depth_far_thres", DEPTH_FAR_THRES, 10.0);
    nh.param<double>("depth_near_thres", DEPTH_NEAR_THRES, 0.3);
    nh.param<double>("loop_cov_pos", loop_cov_pos, 0.013);
    nh.param<double>("loop_cov_ang", loop_cov_ang, 2.5e-04);
    nh.param<int>("min_direction_loop", MIN_DIRECTION_LOOP, 3);
    nh.param<int>("width", width, 400);
    nh.param<int>("height", height, 208);
    int _camconfig;
    nh.param<int>("camera_configuration", _camconfig, 1);
    camera_configuration = (CameraConfig) _camconfig;
    nh.param<std::string>("vins_config_path",vins_config_path, "");
    nh.param<std::string>("pca_comp_path",_pca_comp_path, "");
    nh.param<std::string>("pca_mean_path",_pca_mean_path, "");
    nh.param<std::string>("camera_config_path",camera_config_path, 
        "/home/xuhao/swarm_ws/src/VINS-Fusion-gpu/config/vi_car/cam0_mei.yaml");
    nh.param<std::string>("superpoint_model_path", superpoint_model_path, "");
    nh.param<std::string>("netvlad_model_path", netvlad_model_path, "");
    nh.param<bool>("debug_image", debug_image, false);
    nh.param<std::string>("output_path", OUTPUT_PATH, "");
    
    cv::FileStorage fsSettings;
    fsSettings.open(vins_config_path.c_str(), cv::FileStorage::READ);

    if (camera_configuration == CameraConfig::STEREO_PINHOLE) {
        MAX_DIRS = 1;
    } else if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
        MAX_DIRS = 4;
    } else if (camera_configuration == CameraConfig::PINHOLE_DEPTH) {
        MAX_DIRS = 1;
        fsSettings["depth_topic"] >> DEPTH_TOPIC;
    } else {
        MAX_DIRS = 0;
        ROS_ERROR("[SWARM_LOOP] Camera configuration %d not implement yet.", camera_configuration);
        exit(-1);
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;

    fsSettings["compressed_image0_topic"] >> COMP_IMAGE0_TOPIC;
    fsSettings["compressed_image1_topic"] >> COMP_IMAGE1_TOPIC;

    cv::Mat cv_T;
    fsSettings["body_T_cam0"] >> cv_T;
    Eigen::Matrix4d T;
    cv::cv2eigen(cv_T, T);
    left_extrinsic = toROSPose(Swarm::Pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3)));

    fsSettings["body_T_cam1"] >> cv_T;
    cv::cv2eigen(cv_T, T);

    int is_comp_images = 0;
    fsSettings["is_compressed_images"] >> is_comp_images;

    right_extrinsic = toROSPose(Swarm::Pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3)));

    
    loop_net = new LoopNet(_lcm_uri, send_img, send_whole_img_desc, recv_msg_duration);
    loop_cam = new LoopCam(camera_configuration, camera_config_path, superpoint_model_path, _pca_comp_path, _pca_mean_path, 
        superpoint_thres, superpoint_max_num, netvlad_model_path, width, height, self_id, send_img, nh);
        
    loop_cam->show = debug_image; 
    loop_detector = new LoopDetector(self_id);
    loop_detector->loop_cam = loop_cam;
    loop_detector->enable_visualize = debug_image;

    loop_detector->on_loop_cb = [&] (LoopEdge & loop_con) {
        this->on_loop_connection(loop_con, true);
    };

    loop_net->frame_desc_callback = [&] (const FisheyeFrameDescriptor_t & frame_desc) {
        if (received_image) {
            if (enable_pub_remote_frame) {
                remote_image_desc_pub.publish(toROSFisheyeDescriptor(frame_desc));
            }
            this->on_remote_image(frame_desc);
            this->pub_node_frame(frame_desc);
        }
    };

    loop_net->loopconn_callback = [&] (const LoopEdge_t & loop_conn) {
        auto loc = toROSLoopEdge(loop_conn);
        on_loop_connection(loc, false);
    };

    if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
        flatten_raw_sub = nh.subscribe("/vins_estimator/flattened_gray", 1, &SwarmLoop::flatten_raw_callback, this, ros::TransportHints().tcpNoDelay());
    } else if (camera_configuration == CameraConfig::STEREO_PINHOLE) {
        //Subscribe stereo pinhole, probrably is 
        if (is_comp_images) {
            ROS_INFO("[SWARM_LOOP] Input: compressed images %s and %s", COMP_IMAGE0_TOPIC.c_str(), COMP_IMAGE1_TOPIC.c_str());
            comp_image_sub_l = new message_filters::Subscriber<sensor_msgs::CompressedImage> (nh, COMP_IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            comp_image_sub_r = new message_filters::Subscriber<sensor_msgs::CompressedImage> (nh, COMP_IMAGE1_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            comp_sync = new message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> (*comp_image_sub_l, *comp_image_sub_r, 1000);
            comp_sync->registerCallback(boost::bind(&SwarmLoop::comp_stereo_images_callback, this, _1, _2));
        } else {
            ROS_INFO("[SWARM_LOOP] Input: raw images %s and %s", IMAGE0_TOPIC.c_str(), IMAGE1_TOPIC.c_str());
            image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (nh, IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (nh, IMAGE1_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 1000);
            sync->registerCallback(boost::bind(&SwarmLoop::stereo_images_callback, this, _1, _2));
        }
    } else if (camera_configuration == CameraConfig::PINHOLE_DEPTH) {
        if (is_comp_images) {
            ROS_INFO("[SWARM_LOOP] Input: compressed images %s and depth %s", COMP_IMAGE0_TOPIC.c_str(), DEPTH_TOPIC.c_str());
            comp_image_sub_l = new message_filters::Subscriber<sensor_msgs::CompressedImage> (nh, COMP_IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (nh, DEPTH_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            comp_depth_sync = new message_filters::TimeSynchronizer<sensor_msgs::CompressedImage, sensor_msgs::Image> (*comp_image_sub_l, *image_sub_r, 1000);
            comp_depth_sync->registerCallback(boost::bind(&SwarmLoop::comp_depth_images_callback, this, _1, _2));
        } else {
            ROS_INFO("[SWARM_LOOP] Input: raw images %s and depth %s", IMAGE0_TOPIC.c_str(), DEPTH_TOPIC.c_str());
            image_sub_l = new message_filters::Subscriber<sensor_msgs::Image> (nh, IMAGE0_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            image_sub_r = new message_filters::Subscriber<sensor_msgs::Image> (nh, DEPTH_TOPIC, 1000, ros::TransportHints().tcpNoDelay(true));
            sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*image_sub_l, *image_sub_r, 1000);
            sync->registerCallback(boost::bind(&SwarmLoop::depth_images_callback, this, _1, _2));
        }
    }
    
    keyframe_pub = nh.advertise<swarm_msgs::node_frame>("keyframe", 10);
    odometry_sub  = nh.subscribe("/vins_estimator/odometry", 1, &SwarmLoop::odometry_callback, this, ros::TransportHints().tcpNoDelay());
    keyframe_odometry_sub  = nh.subscribe("/vins_estimator/keyframe_pose", 1, &SwarmLoop::odometry_keyframe_callback, this, ros::TransportHints().tcpNoDelay());

    loopconn_pub = nh.advertise<swarm_msgs::LoopEdge>("loop_connection", 10);
    
    if (enable_sub_remote_frame) {
        ROS_INFO("[SWARM_LOOP] Subscribing remote image from bag");
        remote_img_sub = nh.subscribe("/swarm_loop/remote_frame_desc", 1, &SwarmLoop::on_remote_frame_ros, this, ros::TransportHints().tcpNoDelay());
    }

    if (enable_pub_remote_frame) {
        remote_image_desc_pub = nh.advertise<swarm_msgs::FisheyeFrameDescriptor>("remote_frame_desc", 10);
    }

    if (enable_pub_local_frame) {
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
    