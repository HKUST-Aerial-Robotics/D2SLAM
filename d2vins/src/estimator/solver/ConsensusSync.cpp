#include "ConsensusSync.hpp"

#include <swarm_msgs/swarm_lcm_converter.hpp>

namespace D2VINS {
DistributedVinsData::DistributedVinsData(const DistributedVinsData_t& msg)
    : stamp(toROSTime(msg.timestamp).toSec()),
      drone_id(msg.drone_id),
      solver_token(msg.solver_token),
      iteration_count(msg.iteration_count),
      reference_frame_id(msg.reference_frame_id) {
  for (unsigned int i = 0; i < msg.frame_ids.size(); i++) {
    frame_ids.emplace_back(msg.frame_ids[i]);
    frame_poses.emplace_back(Swarm::Pose(msg.frame_poses[i]));
  }
  for (unsigned int i = 0; i < msg.extrinsic.size(); i++) {
    extrinsic.emplace_back(Swarm::Pose(msg.extrinsic[i]));
    cam_ids.emplace_back(msg.cam_ids[i]);
  }
}

DistributedVinsData_t DistributedVinsData::toLCM() const {
  DistributedVinsData_t msg;
  msg.timestamp = toLCMTime(ros::Time(stamp));
  msg.drone_id = drone_id;
  for (unsigned int i = 0; i < frame_ids.size(); i++) {
    msg.frame_ids.emplace_back(frame_ids[i]);
    msg.frame_poses.emplace_back(frame_poses[i].toLCM());
  }
  for (unsigned int i = 0; i < extrinsic.size(); i++) {
    msg.extrinsic.emplace_back(extrinsic[i].toLCM());
    msg.cam_ids.emplace_back(cam_ids[i]);
  }
  msg.camera_num = extrinsic.size();
  msg.sld_win_len = frame_ids.size();
  msg.solver_token = solver_token;
  msg.iteration_count = iteration_count;
  msg.reference_frame_id = reference_frame_id;
  return msg;
}

}  // namespace D2VINS