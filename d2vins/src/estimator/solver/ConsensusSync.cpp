#include "ConsensusSync.hpp"
#include <swarm_msgs/swarm_lcm_converter.hpp>

namespace D2VINS {
    
DistributedVinsData::DistributedVinsData(const DistributedVinsData_t & msg):
    stamp(toROSTime(msg.timestamp).toSec()), drone_id(msg.drone_id), solver_token(msg.solver_token),
    iteration_count(msg.iteration_count)
{
    for (int i = 0; i < msg.frame_ids.size(); i++) {
        frame_ids.emplace_back(msg.frame_ids[i]);
        frame_poses.emplace_back(Swarm::Pose(msg.frame_poses[i]));
    }
    for (int i = 0; i < msg.extrinsic.size(); i++) {
        extrinsic.emplace_back(Swarm::Pose(msg.extrinsic[i]));
        cam_ids.emplace_back(msg.cam_ids[i]);
    }
    for (int i = 0; i < msg.remote_drone_ids.size(); i++) {
        relative_coordinates.emplace_back(Swarm::Pose(msg.relative_coordinates[i]));
        remote_drone_ids.emplace_back(msg.remote_drone_ids[i]);
    }
}

DistributedVinsData_t DistributedVinsData::toLCM() const {
    DistributedVinsData_t msg;
    msg.timestamp = toLCMTime(ros::Time(stamp));
    msg.drone_id = drone_id;
    for (int i = 0; i < frame_ids.size(); i++) {
        msg.frame_ids.emplace_back(frame_ids[i]);
        msg.frame_poses.emplace_back(fromPose(frame_poses[i]));
    }
    for (int i = 0; i < extrinsic.size(); i++) {
        msg.extrinsic.emplace_back(fromPose(extrinsic[i]));
        msg.cam_ids.emplace_back(cam_ids[i]);
    }
    for (int i = 0; i < relative_coordinates.size(); i++) {
        msg.relative_coordinates.emplace_back(fromPose(relative_coordinates[i]));
        msg.remote_drone_ids.emplace_back(remote_drone_ids[i]);
    }
    msg.camera_num = extrinsic.size();
    msg.sld_win_len = frame_ids.size();
    msg.remote_drone_num = remote_drone_ids.size();
    msg.solver_token = solver_token;
    msg.iteration_count = iteration_count;
    return msg;
}

void SyncDataReceiver::add(const DistributedVinsData & data) {
    const Guard lock(sync_data_recv_lock);
    sync_datas.emplace_back(data);
}

std::vector<DistributedVinsData> SyncDataReceiver::retrive(int64_t token, int iteration_count) {
    const Guard lock(sync_data_recv_lock);
    std::vector<DistributedVinsData> datas;
    for (auto it = sync_datas.begin(); it != sync_datas.end(); ) {
        if (it->solver_token == token && it->iteration_count == iteration_count) {
            datas.emplace_back(*it);
            it = sync_datas.erase(it);
        } else {
            it++;
        }
    }
    return datas;
}

std::vector<DistributedVinsData> SyncDataReceiver::retrive_all() {
    const Guard lock(sync_data_recv_lock);
    std::vector<DistributedVinsData> datas = sync_datas;
    sync_datas.clear();
    return datas;
}

}