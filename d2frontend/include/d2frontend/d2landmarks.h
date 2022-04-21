#pragma once
#include <ros/ros.h>
#include <swarm_msgs/Landmark.h>
#include <swarm_msgs/lcm_gen/Landmark_t.hpp>

namespace D2FrontEnd {

typedef uint64_t FrameIdType;
typedef int LandmarkIdType;

enum LandmarkFlag {
    UNINITIALIZED = 0,
    INITIALIZED = 1, //Initialized by stereo
    ESTIMATED = 2, //ESTIMATE
    OUTLIER=3
};

struct LandmarkPerFrame {
    FrameIdType frame_id = -1;
    LandmarkIdType landmark_id = -1;
    int camera_id = 0;
    int drone_id = -1;
    LandmarkFlag flag = UNINITIALIZED;
    cv::Point2f pt2d;
    Eigen::Vector2d pt2d_norm;
    Eigen::Vector3d pt3d;  //Note this is initialized by frontend and will not be modified by estimator.
    Eigen::Vector3d velocity;
    double depth = -1;
    double cur_td = 0.0;
    bool depth_mea = false;

    void setLandmarkId(LandmarkIdType id) {
        landmark_id = id;
    }

    LandmarkPerFrame(): pt2d_norm(0., 0.), pt3d(0., 0., 0.), velocity(0., 0., 0.)
    {}

    LandmarkPerFrame(const Landmark_t & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.landmark_id),
        camera_id(Landmark.camera_id),
        drone_id(Landmark.drone_id),
        flag((LandmarkFlag) Landmark.flag),
        pt2d(Landmark.pt2d.x, Landmark.pt2d.y),
        pt2d_norm(Landmark.pt2d_norm.x, Landmark.pt2d_norm.y),
        pt3d(Landmark.pt3d.x, Landmark.pt3d.y, Landmark.pt3d.z),
        velocity(Landmark.velocity.x, Landmark.velocity.y, Landmark.velocity.z),
        depth(Landmark.depth)
    {}

    LandmarkPerFrame(const swarm_msgs::Landmark & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.landmark_id),
        camera_id(Landmark.camera_id),
        drone_id(Landmark.drone_id),
        flag((LandmarkFlag) Landmark.flag),
        pt2d(Landmark.pt2d.x, Landmark.pt2d.y),
        pt2d_norm(Landmark.pt2d_norm.x, Landmark.pt2d_norm.y),
        pt3d(Landmark.pt3d.x, Landmark.pt3d.y, Landmark.pt3d.z),
        velocity(Landmark.velocity.x, Landmark.velocity.y, Landmark.velocity.z),
        depth(Landmark.depth)
    {}

    Landmark_t toLCM() {
        Landmark_t ret;
        ret.frame_id = frame_id;
        ret.landmark_id = landmark_id;
        ret.camera_id = camera_id;
        ret.drone_id = drone_id;
        ret.flag = flag;
        ret.depth = depth;
        ret.pt2d.x = pt2d.x;
        ret.pt2d.y = pt2d.y;
        ret.velocity.x = velocity.x();
        ret.velocity.y = velocity.y();
        ret.velocity.z = velocity.z();
        ret.pt2d_norm.x = pt2d_norm.x();
        ret.pt2d_norm.y = pt2d_norm.y();
        ret.pt3d.x = pt3d.x();
        ret.pt3d.y = pt3d.y();
        ret.pt3d.z = pt3d.z();
        return ret;
    }

    swarm_msgs::Landmark toROS() {
        swarm_msgs::Landmark ret;
        ret.frame_id = frame_id;
        ret.landmark_id = landmark_id;
        ret.camera_id = camera_id;
        ret.drone_id = drone_id;
        ret.flag = flag;
        ret.depth = depth;
        ret.pt2d.x = pt2d.x;
        ret.pt2d.y = pt2d.y;
        ret.pt2d.z = 0.0;
        ret.velocity.x = velocity.x();
        ret.velocity.y = velocity.y();
        ret.velocity.z = velocity.z();
        ret.pt2d_norm.x = pt2d_norm.x();
        ret.pt2d_norm.y = pt2d_norm.y();
        ret.pt2d_norm.z = 0.0;
        ret.pt3d.x = pt3d.x();
        ret.pt3d.y = pt3d.y();
        ret.pt3d.z = pt3d.z();
        return ret;
    }

    Vector3d measurement() {
        return Vector3d(pt2d_norm.x(), pt2d_norm.y(), 1.0);
    }
};

struct LandmarkPerId {
    int landmark_id = -1;
    int drone_id = -1;
    std::vector<LandmarkPerFrame> track;
    Eigen::Vector3d position;  //Note thiswill be modified by estimator.
    LandmarkFlag flag;
    LandmarkPerId() {}
    LandmarkPerId(const LandmarkPerFrame & Landmark):
        landmark_id(Landmark.landmark_id),
        drone_id(Landmark.drone_id),
        position(Landmark.pt3d),
        flag(Landmark.flag)
    {
        track.emplace_back(Landmark);
    }

    size_t popFrame(FrameIdType frame_id) {
        for (size_t i = track.size() - 1; i >= 0; i--) {
            if (track[i].frame_id == frame_id) {
                track.erase(track.begin() + i);
                break;
            }
        }
        return track.size();
    }

    void add(const LandmarkPerFrame & Landmark) {
        track.emplace_back(Landmark);
    }
};

}