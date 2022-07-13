#include "d2comm.h"
namespace D2Comm {
void D2Comm::init(ros::NodeHandle & nh) {
    std::string lcm_uri;
    nh.param<std::string>("lcm_uri", lcm_uri, "udpm://224.0.0.251:7667?ttl=1");
    nh.param<int>("self_id", self_id, 0);
    lcm = new lcm::LCM();
    if (!lcm->good()) {
        ROS_ERROR("D2Comm: Failed to initialize LCM.");
        return;
    } else {
        ROS_INFO("D2Comm: LCM initialized at drone %d.", self_id);
    }

    lcm->subscribe("PGO_Sync_Data", &D2Comm::PGODataLCMCallback, this);
    pgo_data_pub = nh.advertise<swarm_msgs::DPGOData>("/d2pgo/pgo_data", 1);
    pgo_data_sub = nh.subscribe("/d2pgo/pgo_data", 1, &D2Comm::PGODataRosCallback, this);
}

void D2Comm::PGODataLCMCallback(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const DistributedPGOData_t * msg) {
    if (msg->drone_id == self_id) {
        return;
    }
    D2Common::DPGOData data(*msg);
    pgo_data_pub.publish(data.toROS());
}

void D2Comm::PGODataRosCallback(const swarm_msgs::DPGOData & ros_data) {
    if (ros_data.drone_id == self_id) {
        return;
    }
    D2Common::DPGOData data(ros_data);
    auto lcm_data = data.toLCM();
    lcm->publish("PGO_Sync_Data", &lcm_data);
}
}