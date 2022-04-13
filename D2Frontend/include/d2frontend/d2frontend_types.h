#pragma once
#include <ros/ros.h>
#include <swarm_msgs/ImageDescriptor_t.hpp>
#include <swarm_msgs/FisheyeFrameDescriptor_t.hpp>
#include <swarm_msgs/FisheyeFrameDescriptor.h>
#include <swarm_msgs/Pose.h>

namespace D2Frontend {
struct StereoFrame{
    ros::Time stamp;
    int keyframe_id;
    std::vector<cv::Mat> left_images, right_images, depth_images;
    Swarm::Pose pose_drone;
    std::vector<Swarm::Pose> left_extrisincs, right_extrisincs;

    StereoFrame():stamp(0) {

    }

    StereoFrame(ros::Time _stamp, cv::Mat _left_image, cv::Mat _right_image, 
        Swarm::Pose _left_extrinsic, Swarm::Pose _right_extrinsic, int self_id):
        stamp(_stamp)
    {
        left_images.push_back(_left_image);
        right_images.push_back(_right_image);
        left_extrisincs.push_back(_left_extrinsic);
        right_extrisincs.push_back(_right_extrinsic);
        keyframe_id = generate_keyframe_id(_stamp, self_id);

    }

    StereoFrame(ros::Time _stamp, cv::Mat _left_image, cv::Mat _dep_image, 
        Swarm::Pose _left_extrinsic, int self_id):
        stamp(_stamp)
    {
        left_images.push_back(_left_image);
        depth_images.push_back(_dep_image);
        left_extrisincs.push_back(_left_extrinsic);
        keyframe_id = generate_keyframe_id(_stamp, self_id);
    }

    // StereoFrame(vins::FlattenImages vins_flatten, int self_id):
    //     stamp(vins_flatten.header.stamp) {
    //     for (int i = 1; i < vins_flatten.up_cams.size(); i++) {
    //         left_extrisincs.push_back(vins_flatten.extrinsic_up_cams[i]);
    //         right_extrisincs.push_back(vins_flatten.extrinsic_down_cams[i]);
            
    //         auto _l = getImageFromMsg(vins_flatten.up_cams[i]);
    //         auto _r = getImageFromMsg(vins_flatten.down_cams[i]);

    //         left_images.push_back(_l->image);
    //         right_images.push_back(_r->image);
    //     }

    //     keyframe_id = generate_keyframe_id(stamp, self_id);
    // }
};

struct VisualImageDesc {
    //This stands for single image
    ros::Time timestamp;
    StereoFrame * stereo_frame = nullptr;
    cv::Mat raw_image;
    int drone_id = 0;
    uint64_t frame_id = 0; 
    int camera_id = 0; //camera id in stereo_frame
    Swarm::Pose extrinsic; //Camera extrinsic
    Swarm::Pose pose_drone; //IMU propagated pose
    std::vector<Vector3d> landmarks_3d;
    std::vector<Vector2d> landmarks_2d_norm; //normalized 2d 
    std::vector<cv::Point2f> landmarks_2d; //normalized 2d 
    std::vector<uint8_t> landmarks_flag; //0 no 3d, 1 has 3d
    std::vector<int> landmarks_id; //0 no 3d, 1 has 3d

    std::vector<float> image_desc;
    std::vector<float> feature_descriptor;
    bool prevent_adding_db = false;

    std::vector<uint8_t> image; //Buffer to store compressed image.

    int landmark_num() const {
        return landmarks_2d.size();
    }
};

struct VisualImageDescArray {
    int drone_id = 0;
    uint64_t frame_id;
    ros::Time stamp;
    std::vector<VisualImageDesc> images;
    Swarm::Pose pose_drone;
    int landmark_num;
    bool prevent_adding_db;
};

inline VisualImageDescArray toVisualImageDescArray(const swarm_msgs::FisheyeFrameDescriptor & img_desc) {
    // return _img;
    VisualImageDescArray fisheye_frame;
    fisheye_frame.frame_id = img_desc.msg_id;
    fisheye_frame.prevent_adding_db = img_desc.prevent_adding_db;
    fisheye_frame.landmark_num = img_desc.landmark_num;
    fisheye_frame.drone_id = img_desc.drone_id;
    fisheye_frame.stamp = img_desc.header.stamp;
    fisheye_frame.pose_drone = Swarm::Pose(img_desc.pose_drone);
    for (auto & _img: img_desc.images) {
        //TODO:
        // fisheye_frame.images.push_back(toLCMImageDescriptor(_img));
    }
    return fisheye_frame;
}
}