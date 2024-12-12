#pragma once
#include <ros/ros.h>
#include <swarm_msgs/Landmark.h>
#include <swarm_msgs/lcm_gen/Landmark_t.hpp>
#include <opencv2/opencv.hpp>
#include "d2basetypes.h"
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include <spdlog/spdlog.h>

namespace D2Common {
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

enum LandmarkType {
    SuperPointLandmark, //Landmark track by superpoint
    FlowLandmark // Landmark track by lk optical flow
};

struct LandmarkPerFrame {
    FrameIdType frame_id = -1;
    LandmarkIdType landmark_id = -1;
    LandmarkType type = LandmarkType::SuperPointLandmark;
    double stamp = 0.0;
    double stamp_discover = 0.0; // first discovery time
    int camera_index = 0;
    int camera_id = 0;
    int drone_id = -1; //-1 is intra landmark
    int solver_id = -1;
    LandmarkFlag flag = UNINITIALIZED;
    cv::Point2f pt2d = cv::Point2f(0, 0);
    Eigen::Vector3d pt3d_norm = Eigen::Vector3d::Zero(); //[x, y, 1]
    Eigen::Vector3d pt3d = Eigen::Vector3d::Zero();  //Note this is initialized by frontend in cam frame and will not be modified by estimator.
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    double depth = -1;
    double cur_td = 0.0;
    bool depth_mea = false;
    cv::Vec3b color = cv::Vec3b(0, 0, 0);

    void setLandmarkId(LandmarkIdType id) {
        landmark_id = id;
    }

    static LandmarkPerFrame createLandmarkPerFrame(LandmarkIdType landmark_id, FrameIdType frame_id, double stamp, 
            LandmarkType type, int drone_id, int camera_index, int camera_id, cv::Point2f pt2d, Eigen::Vector3d pt3d_norm)
    {
        LandmarkPerFrame lm;
        lm.landmark_id = landmark_id;
        lm.frame_id = frame_id;
        lm.type = type;
        lm.camera_index = camera_index;
        lm.camera_id = camera_id;
        lm.drone_id = drone_id;
        lm.stamp = stamp;
        lm.pt2d = pt2d;
        lm.pt3d_norm = pt3d_norm;
        lm.stamp_discover = stamp;
        return lm;
    }

    LandmarkPerFrame(): pt3d_norm(0., 0., 0.), pt3d(0., 0., 0.), velocity(0., 0., 0.)
    {}

    LandmarkPerFrame(const Landmark_t & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.compact.landmark_id),
        type((LandmarkType)Landmark.type),
        stamp(toROSTime(Landmark.timestamp).toSec()),
        stamp_discover(toROSTime(Landmark.compact.stamp_discover).toSec()),
        camera_index(Landmark.camera_index),
        camera_id(Landmark.camera_id),
        drone_id(Landmark.drone_id),
        flag((LandmarkFlag) Landmark.compact.flag),
        pt2d(Landmark.compact.pt2d.x, Landmark.compact.pt2d.y),
        pt3d_norm(Landmark.compact.pt3d_norm.x, Landmark.compact.pt3d_norm.y, Landmark.compact.pt3d_norm.z),
        pt3d(Landmark.compact.pt3d.x, Landmark.compact.pt3d.y, Landmark.compact.pt3d.z),
        velocity(Landmark.compact.velocity.x, Landmark.compact.velocity.y, Landmark.compact.velocity.z),
        depth(Landmark.compact.depth),
        depth_mea(Landmark.compact.depth_mea),
        cur_td(Landmark.cur_td)
    {}

    LandmarkPerFrame(const swarm_msgs::Landmark & Landmark):
        frame_id(Landmark.frame_id),
        landmark_id(Landmark.landmark_id),
        type((LandmarkType)Landmark.type),
        stamp(Landmark.header.stamp.toSec()),
        camera_index(Landmark.camera_index),
        camera_id(Landmark.camera_id),
        drone_id(Landmark.drone_id),
        flag((LandmarkFlag) Landmark.flag),
        pt2d(Landmark.pt2d.x, Landmark.pt2d.y),
        pt3d_norm(Landmark.pt3d_norm.x, Landmark.pt3d_norm.y, Landmark.pt3d_norm.z),
        pt3d(Landmark.pt3d.x, Landmark.pt3d.y, Landmark.pt3d.z),
        velocity(Landmark.velocity.x, Landmark.velocity.y, Landmark.velocity.z),
        depth(Landmark.depth),
        depth_mea(Landmark.depth_mea),
        cur_td(Landmark.cur_td)
    {}

    Landmark_t toLCM() {
        Landmark_t ret;
        ret.frame_id = frame_id;
        ret.compact.landmark_id = landmark_id;
        ret.timestamp = toLCMTime(ros::Time(stamp));
        ret.compact.stamp_discover = toLCMTime(ros::Time(stamp_discover));
        ret.camera_index = camera_index;
        ret.drone_id = drone_id;
        ret.type = type;
        ret.camera_id = camera_id;
        ret.compact.flag = flag;
        ret.compact.depth = depth;
        ret.compact.depth_mea = depth_mea;
        ret.compact.pt2d.x = pt2d.x;
        ret.compact.pt2d.y = pt2d.y;
        ret.compact.velocity.x = velocity.x();
        ret.compact.velocity.y = velocity.y();
        ret.compact.velocity.z = velocity.z();
        ret.compact.pt3d_norm.x = pt3d_norm.x();
        ret.compact.pt3d_norm.y = pt3d_norm.y();
        ret.compact.pt3d_norm.z = pt3d_norm.z();
        ret.compact.pt3d.x = pt3d.x();
        ret.compact.pt3d.y = pt3d.y();
        ret.compact.pt3d.z = pt3d.z();
        ret.cur_td = cur_td;
        return ret;
    }

    swarm_msgs::Landmark toROS() {
        swarm_msgs::Landmark ret;
        ret.header.stamp = ros::Time(stamp);
        ret.frame_id = frame_id;
        ret.landmark_id = landmark_id;
        ret.camera_index = camera_index;
        ret.drone_id = drone_id;
        ret.flag = flag;
        ret.type = type;
        ret.camera_id = camera_id;
        ret.depth = depth;
        ret.depth_mea = depth_mea;
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
        ret.cur_td = cur_td;
        return ret;
    }

    Vector3d measurement() const {
        return pt3d_norm;
    }
};

struct LandmarkPerId {
    int landmark_id = -1;
    int drone_id = -1;
    int base_frame_id = -1;
    int solver_id = -1; //solve on local drone
    double stamp_discover = 0.0; // first discovery time
    std::vector<LandmarkPerFrame> track;
    Eigen::Vector3d position;  //Note thiswill be modified by estimator.
    LandmarkFlag flag = UNINITIALIZED;
    LandmarkSolverFlag solver_flag = UNSOLVED; //If 1, is solved
    cv::Vec3b color;
    int num_outlier_tracks = 0;
    LandmarkPerId():
        color(0, 0, 0) {}
    LandmarkPerId(const LandmarkPerFrame & Landmark):
        landmark_id(Landmark.landmark_id),
        drone_id(Landmark.drone_id),
        position(Landmark.pt3d),
        flag(Landmark.flag),
        color(Landmark.color),
        solver_id(Landmark.solver_id),
        stamp_discover(Landmark.stamp_discover)
    {
        base_frame_id = Landmark.frame_id;
        add(Landmark);
    }

    size_t popBaseFrame() {
        if (track.size() == 0) {
            return 0;
        }
        return popFrame(base_frame_id);
    }

    size_t popFrame(FrameIdType frame_id) {
        if (track.size() == 0) {
            return 0;
        }

        for (auto it=track.begin(); it!=track.end();) {
            if (it->frame_id == frame_id) {
                it = track.erase(it);
            } else {
                ++it;
            }
        }
        if (track.size() > 0) {
            base_frame_id = track.front().frame_id;
        }
        return track.size();
    }

    void add(const LandmarkPerFrame & Landmark) {
        track.emplace_back(Landmark);
        if (Landmark.solver_id >= 0) {
            //Update the solver id
            solver_id = Landmark.solver_id;
        }
    }

    bool shouldBeSolve(int self_id) const {
        if (solver_id == -1 && drone_id != self_id) {
            // This is a internal only remote landmark
            return false;
        }
        if (solver_id > 0 && solver_id != self_id) {
            return false;
        }
        return true;
    }

    double scoreForSolve(int self_id) const {
        if (!shouldBeSolve(self_id)) {
            return -1;
        }
        double score = 0;
        for (auto & it: track) {
            if (it.drone_id != self_id) {
                score += 2;
            } else {
                score += 1;
            }
        }
        return score - num_outlier_tracks*2;  
    }

    bool isMultiCamera() const {
        std::set<int> cam_set;
        for (auto & it: track) {
            cam_set.insert(it.camera_index);
        }
        return cam_set.size() > 1;
    }

    LandmarkPerFrame at(FrameIdType frame_id) const {
        for (auto & it: track) {
            if (it.frame_id == frame_id) {
                return it;
            }
        }
        spdlog::warn("LandmarkPerId::at Warn, cannot find frame_id {}", frame_id);
        return LandmarkPerFrame();
    }

    bool HasFrame(FrameIdType frame_id) const {
        for (auto & it: track) {
            if (it.frame_id == frame_id) {
                return true;
            }
        }
        return false;
    }
};

}