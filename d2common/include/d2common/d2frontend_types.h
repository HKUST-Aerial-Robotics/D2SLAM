#pragma once
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include "d2landmarks.h"
#include "d2imu.h"

namespace D2Common {
inline FrameIdType generateKeyframeId(ros::Time stamp, int self_id) {
    static int keyframe_count = 0;
    int t_ms = 0;//stamp.toSec()*1000;
    return (t_ms%100000)*10000 + self_id*1000000 + keyframe_count++;
}

inline CamIdType generateCameraId(int self_id, int index) {
    return self_id*1000 + index;
}

struct StereoFrame {
    ros::Time stamp;
    int keyframe_id;
    std::vector<cv::Mat> left_images, right_images, depth_images;
    Swarm::Pose pose_drone;
    std::vector<Swarm::Pose> left_extrisincs, right_extrisincs;
    std::vector<int> left_camera_indices;
    std::vector<int> right_camera_indices;
    std::vector<CamIdType> left_camera_ids;
    std::vector<CamIdType> right_camera_ids;

    StereoFrame():stamp(0) {

    }

    StereoFrame(ros::Time _stamp, cv::Mat _left_image, cv::Mat _right_image, 
        Swarm::Pose _left_extrinsic, Swarm::Pose _right_extrinsic, int self_id):
        stamp(_stamp),
        left_images{_left_image},
        right_images{_right_image},
        left_extrisincs{_left_extrinsic},
        right_extrisincs{_right_extrinsic},
        left_camera_indices{0},
        right_camera_indices{1},
        left_camera_ids{generateCameraId(self_id, 0)},
        right_camera_ids{generateCameraId(self_id, 1)}
    {
        keyframe_id = generateKeyframeId(_stamp, self_id);
    }

    StereoFrame(ros::Time _stamp, cv::Mat _left_image, cv::Mat _dep_image, 
        Swarm::Pose _left_extrinsic, int self_id):
        stamp(_stamp),
        left_images{_left_image},
        depth_images{_dep_image},
        left_extrisincs{_left_extrinsic},
        left_camera_indices{0},
        left_camera_ids{generateCameraId(self_id, 0)}
    {
        keyframe_id = generateKeyframeId(_stamp, self_id);
    }
};

struct VisualImageDesc {
    //This stands for single image
    double stamp;
    cv::Mat raw_image;
    cv::Mat raw_depth_image;
    int drone_id = 0;
    FrameIdType frame_id = 0; 
    //The index of view; In stereo. 0 left 1 right + 2 * camera_index are different camera
    //In stereo_fisheye; if use depth on side, top 0 left-back 1-4 down 5
    //If use stereo on side,(top-left-front-right-back) (down-left-front-right-back) 0-10 then.
    int camera_index = 0;
    CamIdType camera_id = 0; //unique id of camera
    Swarm::Pose extrinsic; //Camera extrinsic
    Swarm::Pose pose_drone; //IMU propagated pose
    std::vector<LandmarkPerFrame> landmarks;

    std::vector<float> image_desc;
    std::vector<float> landmark_descriptor;
    std::vector<float> landmark_scores;
    bool prevent_adding_db = false;

    std::vector<uint8_t> image; //Buffer to store compressed image.
    int image_width = 0;
    int image_height = 0;

    int landmarkNum() const {
        return landmarks.size();
    }

    int spLandmarkNum() const {
        int size = 0;
        for (auto& l : landmarks) {
            if (l.type == LandmarkType::SuperPointLandmark) {
                size++;
            }
        }
        return size;
    }
    
    void releaseRawImage() {
        raw_image.release();
        raw_depth_image.release();
    }
    
    VisualImageDesc() {}

    void syncIds(int _drone_id, FrameIdType _frame_id) {
        drone_id = drone_id;
        frame_id = _frame_id;
        for (auto & lm : landmarks) {
            lm.drone_id = drone_id;
            lm.frame_id = frame_id;
        }
    }

    std::vector<cv::Point2f> landmarks2D(bool sp_mode=false, bool normed=false) const {
        std::vector<cv::Point2f> ret;
        for (auto & lm : landmarks) {
            if (sp_mode && lm.type != LandmarkType::SuperPointLandmark) {
                break;
            }
            if (normed) {
                cv::Point2f normed(lm.pt3d_norm.x() / lm.pt3d_norm.z(), lm.pt3d_norm.y() / lm.pt3d_norm.z());
                ret.push_back(normed);
            } else {
                ret.push_back(lm.pt2d);
            }
        }
        return ret;
    }

    swarm_msgs::ImageDescriptor toROS() const {
        swarm_msgs::ImageDescriptor img_desc;
        img_desc.header.stamp = ros::Time(stamp);
        img_desc.drone_id = drone_id;
        img_desc.frame_id = frame_id;
        img_desc.landmark_descriptor = landmark_descriptor;
        img_desc.pose_drone = toROSPose(pose_drone);
        img_desc.camera_extrinsic = toROSPose(extrinsic);
        for (auto landmark: landmarks) {
            img_desc.landmarks.emplace_back(landmark.toROS());
        }
        img_desc.image_desc = image_desc;
        img_desc.image_width = image_width;
        img_desc.image_height = image_height;
        img_desc.image = image;
        img_desc.prevent_adding_db = prevent_adding_db;
        img_desc.camera_index = camera_index;
        img_desc.camera_id = camera_id;
        return img_desc;
    }

    ImageDescriptor_t toLCM() const {
        ImageDescriptor_t img_desc;
        img_desc.timestamp = toLCMTime(ros::Time(stamp));
        img_desc.drone_id = drone_id;
        img_desc.frame_id = frame_id;
        img_desc.landmark_descriptor = landmark_descriptor;
        img_desc.landmark_descriptor_size = landmark_descriptor.size();

        img_desc.pose_drone = pose_drone.toLCM();
        img_desc.camera_extrinsic = extrinsic.toLCM();
        img_desc.landmark_num = landmarks.size();
        for (auto landmark: landmarks) {
            img_desc.landmarks.emplace_back(landmark.toLCM());
        }
        img_desc.image_desc = image_desc;
        img_desc.image_desc_size = image_desc.size();

        img_desc.image_width = image_width;
        img_desc.image_height = image_height;
        img_desc.image = image;
        img_desc.image_size = image.size();
        
        img_desc.prevent_adding_db = prevent_adding_db;
        img_desc.camera_index = camera_index;
        img_desc.camera_id = camera_id;
        // printf("Encoding landmark num %d landmark_descriptor_size %d \n", 
        //     img_desc.landmark_num, img_desc.landmark_descriptor_size);
        return img_desc;
    }

    VisualImageDesc(const swarm_msgs::ImageDescriptor & desc):
        extrinsic(desc.camera_extrinsic),
        pose_drone(desc.pose_drone),
        frame_id(desc.frame_id)
    {
        stamp = desc.header.stamp.toSec();
        drone_id = desc.drone_id;
        landmark_descriptor = desc.landmark_descriptor;
        image_desc = desc.image_desc;
        image = desc.image;
        camera_index = desc.camera_index;
        prevent_adding_db = desc.prevent_adding_db;
        camera_index = desc.camera_index;
        camera_id = desc.camera_id;
        for (auto landmark: desc.landmarks) {
            landmarks.emplace_back(landmark);
        }
    }

    VisualImageDesc(const ImageDescriptor_t & desc):
        extrinsic(desc.camera_extrinsic),
        pose_drone(desc.pose_drone),
        frame_id(desc.frame_id)
    {
        stamp = toROSTime(desc.timestamp).toSec();
        drone_id = desc.drone_id;
        landmark_descriptor = desc.landmark_descriptor;
        image_desc = desc.image_desc;
        image = desc.image;
        camera_index = desc.camera_index;
        prevent_adding_db = desc.prevent_adding_db;
        camera_index = desc.camera_index;
        camera_id = desc.camera_id;
        for (auto landmark: desc.landmarks) {
            landmarks.emplace_back(landmark);
        }
    }

};

struct VisualImageDescArray {
    int drone_id = 0;
    int reference_frame_id = -1;
    FrameIdType frame_id = -1;
    double stamp;
    std::vector<VisualImageDesc> images;
    std::vector<FrameIdType> sld_win_status;
    Swarm::Pose pose_drone;
    bool prevent_adding_db;
    bool is_keyframe = false;
    IMUBuffer imu_buf;
    Vector3d Ba;
    Vector3d Bg;

    void sync_landmark_ids() {
        for (auto & image : images) {
            image.syncIds(drone_id, frame_id);
        }
    }
    
    int landmarkNum() const {
        int num = 0;
        for (auto & img : images) {
            num += img.landmarkNum();
        }
        return num;
    }
    
     int spLandmarkNum() const {
        int num = 0;
        for (auto & img : images) {
            num += img.spLandmarkNum();
        }
        return num;
    }

    void releaseRawImages() {
        for (auto & img : images) {
            img.releaseRawImage();
        }
    }

    VisualImageDescArray() {}
    
    VisualImageDescArray(const swarm_msgs::ImageArrayDescriptor & img_desc):
        frame_id(img_desc.frame_id),
        prevent_adding_db(img_desc.prevent_adding_db),
        drone_id(img_desc.drone_id),
        reference_frame_id(img_desc.reference_frame_id),
        is_keyframe(img_desc.is_keyframe)
    {
        stamp = img_desc.header.stamp.toSec();
        pose_drone = Swarm::Pose(img_desc.pose_drone);
        for (auto & _img: img_desc.images) {
            images.emplace_back(_img);
        }
    }

    VisualImageDescArray(const ImageArrayDescriptor_t & img_desc):
        frame_id(img_desc.frame_id),
        prevent_adding_db(img_desc.prevent_adding_db),
        drone_id(img_desc.drone_id),
        reference_frame_id(img_desc.reference_frame_id),
        is_keyframe(img_desc.is_keyframe),
        Ba(img_desc.Ba.x, img_desc.Ba.y, img_desc.Ba.z),
        Bg(img_desc.Bg.x, img_desc.Bg.y, img_desc.Bg.z),
        sld_win_status(img_desc.sld_win_status.frame_ids)
    {
        stamp = toROSTime(img_desc.timestamp).toSec();
        pose_drone = Swarm::Pose(img_desc.pose_drone);
        for (auto & _img: img_desc.images) {
            images.emplace_back(_img);
        }
        for (unsigned int i = 0; i < img_desc.imu_buf.size(); i ++) {
            auto data = img_desc.imu_buf[i];
            imu_buf.add(IMUData(data));
        }
    }

    swarm_msgs::ImageArrayDescriptor toROS() const {
        swarm_msgs::ImageArrayDescriptor ret;
        ret.frame_id = frame_id;
        ret.prevent_adding_db = prevent_adding_db;
        ret.is_keyframe = is_keyframe;
        ret.drone_id = drone_id;
        ret.header.stamp = ros::Time(stamp);
        ret.landmark_num = landmarkNum();
        ret.pose_drone = pose_drone.toROS();
        ret.reference_frame_id = reference_frame_id;
        for (auto & _img: images) {
            ret.images.emplace_back(_img.toROS());
        }
        return ret;
    }

    ImageArrayDescriptor_t toLCM() const {
        ImageArrayDescriptor_t ret;
        ret.msg_id = frame_id;
        ret.frame_id = frame_id;
        ret.prevent_adding_db = prevent_adding_db;
        ret.drone_id = drone_id;
        ret.timestamp = toLCMTime(ros::Time(stamp));
        ret.landmark_num = landmarkNum();
        ret.pose_drone = pose_drone.toLCM();
        ret.is_keyframe = is_keyframe;
        for (auto & _img: images) {
            ret.images.emplace_back(_img.toLCM());
        }
        ret.image_num = images.size();
        ret.imu_buf_size = imu_buf.size();
        ret.Ba.x = Ba.x();
        ret.Ba.y = Ba.y();
        ret.Ba.z = Ba.z();
        ret.Bg.x = Bg.x();
        ret.Bg.y = Bg.y();
        ret.Bg.z = Bg.z();
        ret.sld_win_status.sld_win_len = sld_win_status.size();
        ret.sld_win_status.timestamp = ret.timestamp;
        ret.sld_win_status.drone_id = ret.drone_id;
        ret.sld_win_status.frame_ids = sld_win_status;
        ret.reference_frame_id = reference_frame_id;
        for (unsigned int i = 0; i < imu_buf.size(); i ++) {
            ret.imu_buf.emplace_back(imu_buf[i].toLCM());
        }
        return ret;
    }
};

}