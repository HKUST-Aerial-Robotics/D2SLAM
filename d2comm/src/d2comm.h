#pragma once
#include <lcm/lcm-cpp.hpp>
#include <ros/ros.h>
#include <d2common/d2pgo_types.h>
#include <thread>

namespace D2Comm {
class D2Comm {
    lcm::LCM * lcm = nullptr;
    void PGODataLCMCallback(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const DistributedPGOData_t * msg);
    void PGODataRosCallback(const swarm_msgs::DPGOData & data);
    ros::Subscriber pgo_data_sub;
    ros::Publisher pgo_data_pub;
    int self_id = 0;
    std::thread th;
public:
    D2Comm() {}
    void init(ros::NodeHandle & nh);
    int lcmHandle() {
        return lcm->handle();
    }

};
}