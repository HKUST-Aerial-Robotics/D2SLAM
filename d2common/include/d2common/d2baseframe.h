#pragma once
#include "utils.hpp"
#include <swarm_msgs/Odometry.h>
#include <swarm_msgs/VIOFrame.h>

namespace D2Common {
struct D2BaseFrame {
    double stamp = 0;
    FrameIdType frame_id = -1;
    int drone_id = -1;
    int reference_frame_id = -1; //For which the frame is reference at. Initially, this should be drone_id. After map merge, this should be main id.
    bool is_keyframe = false;
    Swarm::Odometry odom;
    Swarm::Pose initial_ego_pose; //Only effective if this keyframe is from remote
    D2BaseFrame() {}
    D2BaseFrame(double _stamp, FrameIdType _frame_id, int _drone_id, int _reference_frame_id, bool _is_keyframe, Swarm::Odometry _odom, Swarm::Pose _initial_ego_pose):
        stamp(_stamp),
        frame_id(_frame_id),
        drone_id(_drone_id),
        reference_frame_id(_reference_frame_id),
        is_keyframe(_is_keyframe),
        odom(_odom),
        initial_ego_pose(_initial_ego_pose) {}
    D2BaseFrame(double _stamp, FrameIdType _frame_id, int _drone_id, int _reference_frame_id, bool _is_keyframe, Swarm::Odometry _odom):
        stamp(_stamp),
        frame_id(_frame_id),
        drone_id(_drone_id),
        reference_frame_id(_reference_frame_id),
        is_keyframe(_is_keyframe),
        odom(_odom),
        initial_ego_pose(_odom.pose()) {}
    D2BaseFrame(double _stamp, FrameIdType _frame_id, int _drone_id, int _reference_frame_id, bool _is_keyframe, Swarm::Pose pose):
        stamp(_stamp),
        frame_id(_frame_id),
        drone_id(_drone_id),
        reference_frame_id(_reference_frame_id),
        is_keyframe(_is_keyframe),
        odom(_stamp, pose),
        initial_ego_pose(pose) { }

    D2BaseFrame(const swarm_msgs::VIOFrame & vio_frame):
        stamp(vio_frame.header.stamp.toSec()),
        frame_id(vio_frame.frame_id),
        drone_id(vio_frame.drone_id),
        reference_frame_id(vio_frame.reference_frame_id),
        is_keyframe(vio_frame.is_keyframe),
        odom(vio_frame.odom),
        initial_ego_pose(vio_frame.odom.pose.pose) {}
    
    virtual void moveByPose(int new_ref_frame_id, const Swarm::Pose & delta_pose) {
        reference_frame_id = new_ref_frame_id;
        odom.moveByPose(delta_pose);
    }
    const Matrix3d R() const {
        return odom.R();
    }
    const Vector3d T() const {
        return odom.pos();
    }
};

using D2BaseFramePtr = std::shared_ptr<D2Common::D2BaseFrame>;

} // namespace D2Common
