#include "d2comm.h"
namespace D2Comm {
void D2Comm::Init(ros::NodeHandle & nh) {
    std::string lcm_uri;
    nh.param<std::string>("lcm_uri", lcm_uri, "udpm://224.0.0.251:7667?ttl=1");
    lcm = new lcm::LCM();
    if (!lcm->good()) {
        ROS_ERROR("D2Comm: Failed to initialize LCM.");
        return;
    }
    lcm->subscribe("PGO_Sync_Data", &D2Comm::onPGOSyncData, this);
}

}