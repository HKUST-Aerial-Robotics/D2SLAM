#pragma once
#include <swarm_msgs/Pose.h>
#include <swarm_msgs/lcm_gen/DistributedVinsData_t.hpp>
#include <d2common/d2basetypes.h>
#include <d2common/solver/BaseConsensusSync.hpp>
#include <mutex>

typedef std::lock_guard<std::recursive_mutex> Guard;

using namespace D2Common;

namespace D2VINS {
struct DistributedVinsData {
    double stamp;
    int drone_id;
    int solver_token;
    int iteration_count;
    int reference_frame_id = -1;
    std::vector<FrameIdType> frame_ids;
    std::vector<Swarm::Pose> frame_poses;
    std::vector<CamIdType> cam_ids;
    std::vector<Swarm::Pose> extrinsic;
    DistributedVinsData() {}
    DistributedVinsData(const DistributedVinsData_t & msg);
    DistributedVinsData_t toLCM() const;
};

typedef BaseSyncDataReceiver<DistributedVinsData> SyncDataReceiver;

}