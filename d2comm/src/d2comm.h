#pragma once
#include <lcm/lcm-cpp.hpp>
#include <ros/ros.h>

namespace D2Comm {
class D2Comm {
    lcm::LCM * lcm = nullptr;
public:
    D2Comm() {}
    void init(ros::NodeHandle & nh);
    int lcmHandle() {
        return lcm.handle();
    }

    void onPGOSyncData() {

    }
};
}