#pragma once
#include <swarm_msgs/Pose.h>
#include <swarm_msgs/lcm_gen/DistributedVinsData_t.hpp>
#include <d2common/d2basetypes.h>
#include <mutex>

using namespace D2Common;

typedef std::lock_guard<std::recursive_mutex> Guard;

namespace D2VINS {
struct DistributedVinsData {
    double stamp;
    int drone_id;
    int solver_token;
    int iteration_count;
    std::vector<FrameIdType> frame_ids;
    std::vector<Swarm::Pose> frame_poses;
    std::vector<CamIdType> cam_ids;
    std::vector<Swarm::Pose> extrinsic;
    std::vector<int> remote_drone_ids;
    std::vector<Swarm::Pose> relative_coordinates;
    DistributedVinsData() {}
    DistributedVinsData(const DistributedVinsData_t & msg);
    DistributedVinsData_t toLCM() const;
};

class SyncDataReceiver {
protected:
    std::recursive_mutex sync_data_recv_lock;
    std::vector<DistributedVinsData> sync_datas;
public:
    void add(const DistributedVinsData & data);
    std::vector<DistributedVinsData> retrive(int64_t token, int iteration_count);
    std::vector<DistributedVinsData> retrive_all();
};
}