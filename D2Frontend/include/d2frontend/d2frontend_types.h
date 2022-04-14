#pragma once
#include <ros/ros.h>
#include <swarm_msgs/ImageDescriptor_t.hpp>
#include <swarm_msgs/FisheyeFrameDescriptor_t.hpp>
#include <swarm_msgs/FisheyeFrameDescriptor.h>
#include <swarm_msgs/Pose.h>
#include <swarm_msgs/swarm_lcm_converter.hpp>

namespace D2Frontend {

inline int generate_keyframe_id(ros::Time stamp, int self_id) {
    static int keyframe_count = 0;
    int t_ms = 0;//stamp.toSec()*1000;
    return (t_ms%100000)*10000 + self_id*1000000 + keyframe_count++;
}

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
    ros::Time stamp;
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
    std::vector<int32_t> landmarks_id; //0 no 3d, 1 has 3d

    std::vector<float> image_desc;
    std::vector<float> feature_descriptor;
    bool prevent_adding_db = false;

    std::vector<uint8_t> image; //Buffer to store compressed image.
    int image_width = 0;
    int image_height = 0;

    int landmark_num() const {
        return landmarks_2d.size();
    }
    
    VisualImageDesc() {}

    swarm_msgs::ImageDescriptor toROS() const {
        swarm_msgs::ImageDescriptor img_desc;
        img_desc.header.stamp = stamp;
        img_desc.drone_id = drone_id;
        img_desc.feature_descriptor = feature_descriptor;
        img_desc.pose_drone = toROSPose(pose_drone);
        img_desc.camera_extrinsic = toROSPose(extrinsic);
        img_desc.landmarks_2d_norm = toROSPoints(landmarks_2d_norm);
        img_desc.landmarks_2d = toROSPoints(landmarks_2d);
        img_desc.landmarks_3d = toROSPoints(landmarks_3d);
        img_desc.landmarks_id = landmarks_id;

        img_desc.image_desc = image_desc;
        img_desc.image_width = image_width;
        img_desc.image_height = image_height;
        img_desc.image = image;
        img_desc.prevent_adding_db = prevent_adding_db;
        img_desc.landmarks_flag = landmarks_flag;
        img_desc.direction = camera_id;
        return img_desc;
    }

    ImageDescriptor_t toLCM() const {
        ImageDescriptor_t img_desc;
        img_desc.timestamp = toLCMTime(stamp);
        img_desc.drone_id = drone_id;
        img_desc.feature_descriptor = feature_descriptor;
        img_desc.pose_drone = fromPose(pose_drone);
        img_desc.camera_extrinsic = fromPose(extrinsic);
        CVPoints2LCM(landmarks_2d, img_desc.landmarks_2d);
        img_desc.landmarks_2d_norm = toLCMPoints(landmarks_2d_norm);
        img_desc.landmarks_3d = toLCMPoints(landmarks_3d);
        img_desc.landmarks_id = landmarks_id;

        img_desc.image_desc = image_desc;
        img_desc.image_width = image_width;
        img_desc.image_height = image_height;
        img_desc.image = image;
        img_desc.prevent_adding_db = prevent_adding_db;
        img_desc.landmarks_flag = landmarks_flag;
        img_desc.direction = camera_id;
        return img_desc;
    }

    VisualImageDesc(const swarm_msgs::ImageDescriptor & desc):
        extrinsic(desc.camera_extrinsic),
        pose_drone(desc.pose_drone)
    {
        stamp = desc.header.stamp;
        drone_id = desc.drone_id;
        feature_descriptor = desc.feature_descriptor;
        image_desc = desc.image_desc;
        image = desc.image;
        camera_id = desc.direction;
        landmarks_2d_norm = toEigen(desc.landmarks_2d_norm);
        landmarks_3d = toEigen3d(desc.landmarks_3d);
        landmarks_2d = toCV(desc.landmarks_2d);
        landmarks_flag = desc.landmarks_flag;
        landmarks_id = desc.landmarks_id;
        prevent_adding_db = desc.prevent_adding_db;
    }

    VisualImageDesc(const ImageDescriptor_t & desc):
        extrinsic(desc.camera_extrinsic),
        pose_drone(desc.pose_drone)
    {
        stamp = toROSTime(desc.timestamp);
        drone_id = desc.drone_id;
        feature_descriptor = desc.feature_descriptor;
        image_desc = desc.image_desc;
        image = desc.image;
        camera_id = desc.direction;
        landmarks_2d_norm = toEigen(desc.landmarks_2d_norm);
        landmarks_3d = toEigen3d(desc.landmarks_3d);
        landmarks_2d = toCV(desc.landmarks_2d);
        landmarks_flag = desc.landmarks_flag;
        landmarks_id = desc.landmarks_id;
        prevent_adding_db = desc.prevent_adding_db;
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
    
    VisualImageDescArray() {}
    
    VisualImageDescArray(const swarm_msgs::FisheyeFrameDescriptor & img_desc) {
        frame_id = img_desc.msg_id;
        prevent_adding_db = img_desc.prevent_adding_db;
        drone_id = img_desc.drone_id;
        stamp = img_desc.header.stamp;
        landmark_num = img_desc.landmark_num;
        pose_drone = Swarm::Pose(img_desc.pose_drone);
        for (auto & _img: img_desc.images) {
            images.emplace_back(_img);
        }
    }

    VisualImageDescArray(const FisheyeFrameDescriptor_t & img_desc) {
        frame_id = img_desc.msg_id;
        prevent_adding_db = img_desc.prevent_adding_db;
        drone_id = img_desc.drone_id;
        stamp = toROSTime(img_desc.timestamp);
        landmark_num = img_desc.landmark_num;
        pose_drone = Swarm::Pose(img_desc.pose_drone);
        for (auto & _img: img_desc.images) {
            images.emplace_back(_img);
        }
    }

    swarm_msgs::FisheyeFrameDescriptor toROS() const {
        swarm_msgs::FisheyeFrameDescriptor ret;
        ret.msg_id = frame_id;
        ret.prevent_adding_db = prevent_adding_db;
        ret.drone_id = drone_id;
        ret.header.stamp = stamp;
        ret.landmark_num = landmark_num;
        ret.pose_drone = pose_drone.to_ros_pose();
        for (auto & _img: images) {
            ret.images.emplace_back(_img.toROS());
        }
        return ret;
    }

    FisheyeFrameDescriptor_t toLCM() const {
        FisheyeFrameDescriptor_t ret;
        ret.msg_id = frame_id;
        ret.prevent_adding_db = prevent_adding_db;
        ret.drone_id = drone_id;
        ret.timestamp = toLCMTime(stamp);
        ret.landmark_num = landmark_num;
        ret.pose_drone = fromPose(pose_drone);
        for (auto & _img: images) {
            ret.images.emplace_back(_img.toLCM());
        }
        return ret;
    }
};



}