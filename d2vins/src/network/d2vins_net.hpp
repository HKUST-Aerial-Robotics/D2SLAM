#pragma once
#include <lcm/lcm-cpp.hpp>
#include "../estimator/d2vinsstate.hpp"
#include <functional>
#include <swarm_msgs/lcm_gen/SlidingWindow_t.hpp>

namespace D2VINS {
class D2Estimator;
class D2VINSNet {
    std::function<void(int, double, std::vector<FrameIdType>)> remote_sld_win_callback;
    D2EstimatorState & state;
    D2Estimator * estimator;
    lcm::LCM lcm;
public:
    D2VINSNet(D2Estimator * _estimator);
    void pubSlidingWindow();
    void onSldWinReceived(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const SlidingWindow_t* msg);
    int lcmHandle() {
        return lcm.handle();
    }
};
}