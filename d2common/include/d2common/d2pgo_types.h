#pragma once
#include <swarm_msgs/lcm_gen/DistributedPGOData_t.hpp>
#include <swarm_msgs/DPGOData.h>
#include <swarm_msgs/Pose.h>
#include <d2common/d2basetypes.h>

namespace D2Common {
enum DPGODataType {
    DPGO_POSE_DUAL = 0,
    DPGO_ROT_MAT_DUAL,
    DPGO_DELTA_POSE_DUAL
};

class DPGOData {
public:
    double stamp = 0;
    int drone_id = -1;
    int target_id = -1;
    int reference_frame_id = -1;
    int64_t solver_token = -1;
    int iteration_count = -1;
    DPGODataType type = DPGO_POSE_DUAL;

    std::map<FrameIdType, Swarm::Pose> frame_poses;
    std::map<FrameIdType, VectorXd> frame_duals;

    DPGOData() {}
    DPGOData(const swarm_msgs::DPGOData & msg);
    DPGOData(const DistributedPGOData_t & msg);
    swarm_msgs::DPGOData toROS() const;
    DistributedPGOData_t toLCM() const;

};
}