#pragma once
#include "d2frontend_types.h"
#include "utils.hpp"
#include <swarm_msgs/Odometry.h>
#include "d2imu.h"
#include <swarm_msgs/VIOFrame.h>

namespace D2Common {
class IntegrationBase;
struct VINSFrame {
    double stamp = 0;
    FrameIdType frame_id = -1;
    int drone_id = -1;
    int reference_frame_id = -1; //For which the frame is reference at. Initially, this should be drone_id. After map merge, this should be main id.
    bool is_keyframe = false;
    Swarm::Odometry odom;
    Swarm::Pose initial_ego_pose; //Only effective if this keyframe is from remote
    Vector3d Ba; // bias of acc
    Vector3d Bg; //bias of gyro
    FrameIdType prev_frame_id = -1;
    IntegrationBase * pre_integrations = nullptr;
    int imu_buf_index = 0;
    VINSFrame():Ba(0., 0., 0.), Bg(0., 0., 0.)
    {}
    
    VINSFrame(const VisualImageDescArray & frame, const IMUBuffer & buf, const VINSFrame & prev_frame);
    VINSFrame(const VisualImageDescArray & frame, const std::pair<IMUBuffer, int> & buf, const VINSFrame & prev_frame);
    
    VINSFrame(const VisualImageDescArray & frame, const Vector3d & _Ba, const Vector3d & _Bg);
    VINSFrame(const VisualImageDescArray & frame);

    std::string toStr();
    swarm_msgs::VIOFrame toROS();
    swarm_msgs::VIOFrame toROS(const std::vector<Swarm::Pose> & exts);

    void toVector(state_type * _pose, state_type * _spd_bias) const;

    void fromVector(state_type * _pose, state_type * _spd_bias);

    void moveByPose(int new_ref_frame_id, const Swarm::Pose & delta_pose);
};
   
}