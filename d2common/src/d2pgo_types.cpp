#include <d2common/d2pgo_types.h>
#include <std_msgs/Float64MultiArray.h>

#include <swarm_msgs/swarm_lcm_converter.hpp>

namespace D2Common {
DPGOData::DPGOData(const swarm_msgs::DPGOData& msg) {
  stamp = msg.header.stamp.toSec();
  drone_id = msg.drone_id;
  target_id = msg.target_id;
  reference_frame_id = msg.reference_frame_id;
  type = static_cast<DPGODataType>(msg.type);
  for (size_t i = 0; i < msg.frame_ids.size(); i++) {
    frame_poses[msg.frame_ids[i]] = Swarm::Pose(msg.frame_poses[i]);
    frame_duals[msg.frame_ids[i]] = Map<const VectorXd>(
        msg.frame_duals[i].data.data(), msg.frame_duals[i].data.size());
  }
  solver_token = msg.solver_token;
  iteration_count = msg.iteration_count;
}

DPGOData::DPGOData(const DistributedPGOData_t& msg) {
  stamp = toROSTime(msg.timestamp).toSec();
  drone_id = msg.drone_id;
  target_id = msg.target_id;
  type = static_cast<DPGODataType>(msg.type);
  reference_frame_id = msg.reference_frame_id;
  for (size_t i = 0; i < msg.frame_ids.size(); i++) {
    frame_poses[msg.frame_ids[i]] = Swarm::Pose(msg.frame_poses[i]);
    Map<const VectorXf> dual(msg.frame_duals[i].data.data(),
                             msg.frame_duals[i].data.size());
    frame_duals[msg.frame_ids[i]] = dual.template cast<double>();
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
  msg.type = type;
  for (auto it : frame_poses) {
    auto i = it.first;
    auto pose = it.second;
    msg.frame_poses.emplace_back(pose.toROS());
    msg.frame_ids.emplace_back(i);
    VectorXd dual = frame_duals.at(i);
    // Convert VectorXd of frame_duals to Float64MultiArray
    std_msgs::Float64MultiArray dual_array;
    dual_array.layout.data_offset = 0;
    std_msgs::MultiArrayDimension dim0;
    dual_array.layout.dim.emplace_back(dim0);
    dim0.label = "length";
    dim0.size = dual.size();
    dim0.stride = dual.size();
    dual_array.data =
        std::vector<double>(dual.data(), dual.data() + dual.size());
    msg.frame_duals.emplace_back(dual_array);
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
  msg.type = type;
  for (auto it : frame_poses) {
    auto i = it.first;
    auto pose = it.second;
    msg.frame_poses.emplace_back(pose.toLCM());
    msg.frame_ids.emplace_back(i);
    VectorXd dual = frame_duals.at(i);
    Vector_t dual_vec;
    dual_vec.size = dual.size();
    dual_vec.data.resize(dual.size());
    Map<VectorXf> _dual(dual_vec.data.data(), dual_vec.data.size());
    _dual = dual.template cast<float>();
    msg.frame_duals.emplace_back(dual_vec);
  }
  msg.frame_num = frame_poses.size();
  msg.frame_poses_num = frame_poses.size();
  msg.frame_dual_num = frame_duals.size();
  msg.solver_token = solver_token;
  msg.iteration_count = iteration_count;
  return msg;
}
};  // namespace D2Common