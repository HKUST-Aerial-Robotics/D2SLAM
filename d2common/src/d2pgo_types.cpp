#include <d2common/d2pgo_types.h>
#include <swarm_msgs/swarm_lcm_converter.hpp>

namespace D2Common {
DPGOData::DPGOData(const swarm_msgs::DPGOData & msg) {
    stamp = msg.header.stamp.toSec();
    drone_id = msg.drone_id;
    target_id = msg.target_id;
    reference_frame_id = msg.reference_frame_id;
    for (size_t i = 0; i < msg.frame_ids.size(); i ++ ) {
        frame_poses[msg.frame_ids[i]] = Swarm::Pose(msg.frame_poses[i]);
    }
    solver_token = msg.solver_token;
    iteration_count = msg.iteration_count;
}

DPGOData::DPGOData(const DistributedPGOData_t & msg) {
    stamp = toROSTime(msg.timestamp).toSec();
    drone_id = msg.drone_id;
    target_id = msg.target_id;
    reference_frame_id = msg.reference_frame_id;
    for (size_t i = 0; i < msg.frame_ids.size(); i ++ ) {
        frame_poses[msg.frame_ids[i]] = Swarm::Pose(msg.frame_poses[i]);
    }
    solver_token = msg.solver_token;
    iteration_count = msg.iteration_count;
}

swarm_msgs::DPGOData DPGOData::toROS() const {
    swarm_msgs::DPGOData msg;
    msg.header.stamp = ros::Time(stamp);
    msg.drone_id = drone_id;
    msg.target_id = target_id;
    msg.reference_frame_id = reference_frame_id;
    msg.frame_poses.resize(frame_poses.size());
    for (auto it: frame_poses) {
        auto i = it.first;
        auto pose = it.second;
        msg.frame_poses[i] = pose.toROS();
        msg.frame_ids.push_back(i);
    }
    msg.solver_token = solver_token;
    msg.iteration_count = iteration_count;
    return msg;
}

DistributedPGOData_t DPGOData::toLCM() const {
    DistributedPGOData_t msg;
    msg.timestamp = toLCMTime(ros::Time(stamp));
    msg.drone_id = drone_id;
    msg.target_id = target_id;
    msg.reference_frame_id = reference_frame_id;
    msg.frame_poses.resize(frame_poses.size());
    for (auto it: frame_poses) {
        auto i = it.first;
        auto pose = it.second;
        msg.frame_poses[i] = pose.toLCM();
        msg.frame_ids.push_back(i);
    }
    msg.frame_num = frame_poses.size();
    msg.solver_token = solver_token;
    msg.iteration_count = iteration_count;
    return msg;
}
};