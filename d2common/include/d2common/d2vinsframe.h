#pragma once
#include "d2frontend_types.h"
#include <swarm_msgs/Odometry.h>
#include "d2imu.h"
#include "d2baseframe.h"
#include <swarm_msgs/VIOFrame.h>

namespace D2Common {
class IntegrationBase;

struct VINSFrame: public D2BaseFrame {
    Vector3d Ba; // bias of acc
    Vector3d Bg; //bias of gyro
    FrameIdType prev_frame_id = -1;
    IntegrationBase * pre_integrations = nullptr; // From prev to this
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

    void toVector(const StatePtr& pose, const StatePtr& spd_bias) const;
    void fromVector(const StatePtr& pose, const StatePtr& spd_bias);
    D2BaseFrame toBaseFrame() {
        return D2BaseFrame(stamp, frame_id, drone_id, reference_frame_id, is_keyframe, odom, initial_ego_pose);
    }
};
   
}