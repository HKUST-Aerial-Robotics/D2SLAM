#pragma once
#include <ros/ros.h>
#include <swarm_msgs/Landmark.h>
#include <swarm_msgs/lcm_gen/Landmark_t.hpp>

namespace D2FrontEnd {

typedef uint64_t FrameIdType;
typedef int LandmarkIdType;

struct LandmarkPerFrame {
    FrameIdType frame_id = -1;
    LandmarkIdType landmark_id = -1;
    int drone_id = -1;
    int flag = -1;
    cv::Point2f pt2d;
    Eigen::Vector2d pt2d_norm;
    Eigen::Vector3d pt3d;
    double depth = -1;

    void setLandmarkId(LandmarkIdType id) {
        landmark_id = id;
    }

    LandmarkPerFrame() {}
    LandmarkPerFrame(const Landmark_t & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.landmark_id),
        drone_id(Landmark.drone_id),
        flag(Landmark.flag),
        pt2d(Landmark.pt2d.x, Landmark.pt2d.y),
        pt2d_norm(Landmark.pt2d_norm.x, Landmark.pt2d_norm.y),
        pt3d(Landmark.pt3d.x, Landmark.pt3d.y, Landmark.pt3d.z),
        depth(Landmark.depth)
    {}

    LandmarkPerFrame(const swarm_msgs::Landmark & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.landmark_id),
        drone_id(Landmark.drone_id),
        flag(Landmark.flag),
        pt2d(Landmark.pt2d.x, Landmark.pt2d.y),
        pt2d_norm(Landmark.pt2d_norm.x, Landmark.pt2d_norm.y),
        pt3d(Landmark.pt3d.x, Landmark.pt3d.y, Landmark.pt3d.z),
        depth(Landmark.depth)
    {}

    Landmark_t toLCM() {
        Landmark_t ret;
        ret.frame_id = frame_id;
        ret.landmark_id = landmark_id;
        ret.drone_id = drone_id;
        ret.flag = flag;
        ret.depth = depth;
        ret.pt2d.x = pt2d.x;
        ret.pt2d.y = pt2d.y;
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
        ret.drone_id = drone_id;
        ret.flag = flag;
        ret.depth = depth;
        ret.pt2d.x = pt2d.x;
        ret.pt2d.y = pt2d.y;
        ret.pt2d.z = 0.0;
        ret.pt2d_norm.x = pt2d_norm.x();
        ret.pt2d_norm.y = pt2d_norm.y();
        ret.pt2d_norm.z = 0.0;
        ret.pt3d.x = pt3d.x();
        ret.pt3d.y = pt3d.y();
        ret.pt3d.z = pt3d.z();
        return ret;
    }
};

struct LandmarkPerId {
    FrameIdType frame_id = -1;
    int landmark_id = -1;
    int drone_id = -1;
    std::vector<LandmarkPerFrame> track;
    LandmarkPerId() {}
    LandmarkPerId(const LandmarkPerFrame & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.landmark_id),
        drone_id(Landmark.drone_id) {
        track.emplace_back(Landmark);
    }

    void add(const LandmarkPerFrame & Landmark) {
        track.emplace_back(Landmark);
    }
};

}