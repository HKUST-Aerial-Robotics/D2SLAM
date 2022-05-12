#pragma once
#include <ros/ros.h>
#include <swarm_msgs/Landmark.h>
#include <swarm_msgs/lcm_gen/Landmark_t.hpp>

namespace D2FrontEnd {

typedef uint64_t FrameIdType;
typedef int64_t LandmarkIdType;
typedef int32_t CamIdType;

enum LandmarkFlag {
    UNINITIALIZED = 0,
    INITIALIZED = 1, //Initialized by stereo
    ESTIMATED = 2, //ESTIMATE
    OUTLIER=3
};

enum LandmarkSolverFlag {
    UNSOLVED = 0,
    SOLVED = 1
};

struct LandmarkPerFrame {
    FrameIdType frame_id = -1;
    LandmarkIdType landmark_id = -1;
    int camera_index = 0;
    int camera_id = 0;
    int drone_id = -1;
    LandmarkFlag flag = UNINITIALIZED;
    cv::Point2f pt2d;
    Eigen::Vector3d pt3d_norm; //[x, y, 1]
    Eigen::Vector3d pt3d;  //Note this is initialized by frontend and will not be modified by estimator.
    Eigen::Vector3d velocity;
    double depth = -1;
    double cur_td = 0.0;
    bool depth_mea = false;

    void setLandmarkId(LandmarkIdType id) {
        landmark_id = id;
    }

    LandmarkPerFrame(): pt3d_norm(0., 0., 0.), pt3d(0., 0., 0.), velocity(0., 0., 0.)
    {}

    LandmarkPerFrame(const Landmark_t & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.landmark_id),
        camera_index(Landmark.camera_index),
        drone_id(Landmark.drone_id),
        flag((LandmarkFlag) Landmark.flag),
        pt2d(Landmark.pt2d.x, Landmark.pt2d.y),
        pt3d_norm(Landmark.pt3d_norm.x, Landmark.pt3d_norm.y, Landmark.pt3d_norm.z),
        pt3d(Landmark.pt3d.x, Landmark.pt3d.y, Landmark.pt3d.z),
        velocity(Landmark.velocity.x, Landmark.velocity.y, Landmark.velocity.z),
        depth(Landmark.depth)
    {}

    LandmarkPerFrame(const swarm_msgs::Landmark & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.landmark_id),
        camera_index(Landmark.camera_index),
        drone_id(Landmark.drone_id),
        flag((LandmarkFlag) Landmark.flag),
        pt2d(Landmark.pt2d.x, Landmark.pt2d.y),
        pt3d_norm(Landmark.pt3d_norm.x, Landmark.pt3d_norm.y, Landmark.pt3d_norm.z),
        pt3d(Landmark.pt3d.x, Landmark.pt3d.y, Landmark.pt3d.z),
        velocity(Landmark.velocity.x, Landmark.velocity.y, Landmark.velocity.z),
        depth(Landmark.depth)
    {}

    Landmark_t toLCM() {
        Landmark_t ret;
        ret.frame_id = frame_id;
        ret.landmark_id = landmark_id;
        ret.camera_index = camera_index;
        ret.drone_id = drone_id;
        ret.flag = flag;
        ret.depth = depth;
        ret.pt2d.x = pt2d.x;
        ret.pt2d.y = pt2d.y;
        ret.velocity.x = velocity.x();
        ret.velocity.y = velocity.y();
        ret.velocity.z = velocity.z();
        ret.pt3d_norm.x = pt3d_norm.x();
        ret.pt3d_norm.y = pt3d_norm.y();
        ret.pt3d_norm.z = pt3d_norm.z();
        ret.pt3d.x = pt3d.x();
        ret.pt3d.y = pt3d.y();
        ret.pt3d.z = pt3d.z();
        return ret;
    }

    swarm_msgs::Landmark toROS() {
        swarm_msgs::Landmark ret;
        ret.frame_id = frame_id;
        ret.landmark_id = landmark_id;
        ret.camera_index = camera_index;
        ret.drone_id = drone_id;
        ret.flag = flag;
        ret.depth = depth;
        ret.pt2d.x = pt2d.x;
        ret.pt2d.y = pt2d.y;
        ret.pt2d.z = 0.0;
        ret.velocity.x = velocity.x();
        ret.velocity.y = velocity.y();
        ret.velocity.z = velocity.z();
        ret.pt3d_norm.x = pt3d_norm.x();
        ret.pt3d_norm.y = pt3d_norm.y();
        ret.pt3d_norm.z = pt3d_norm.z();
        ret.pt3d_norm.z = 0.0;
        ret.pt3d.x = pt3d.x();
        ret.pt3d.y = pt3d.y();
        ret.pt3d.z = pt3d.z();
        return ret;
    }

    Vector3d measurement() {
        return Vector3d(pt3d_norm.x(), pt3d_norm.y(), pt3d_norm.z());
    }
};

struct LandmarkPerId {
    int landmark_id = -1;
    int drone_id = -1;
    std::vector<LandmarkPerFrame> track;
    std::vector<LandmarkPerFrame> track_r; // tracks of right camera of that point
    Eigen::Vector3d position;  //Note thiswill be modified by estimator.
    LandmarkFlag flag = UNINITIALIZED;
    LandmarkSolverFlag solver_flag = UNSOLVED; //If 1, is solved
    LandmarkPerId() {}
    LandmarkPerId(const LandmarkPerFrame & Landmark):
        landmark_id(Landmark.landmark_id),
        drone_id(Landmark.drone_id),
        position(Landmark.pt3d),
        flag(Landmark.flag)
    {
        add(Landmark);
    }

    size_t popFrame(FrameIdType frame_id) {
        for (int i = 0; i < track.size(); i ++ ) {
            if (track[i].frame_id == frame_id) {
                track.erase(track.begin() + i);
                break;
            }
        }
        for (int i = 0; i < track_r.size(); i ++ ) {
            if (track_r[i].frame_id == frame_id) {
                track_r.erase(track_r.begin() + i);
                break;
            }
        }
        return track.size();
    }

    void add(const LandmarkPerFrame & Landmark) {
        //Simpified.
        //Need to adopt for omni.
        if (Landmark.camera_index == 0) {
            track.emplace_back(Landmark);
        } else {
            track_r.emplace_back(Landmark);
        }
    }
};

}