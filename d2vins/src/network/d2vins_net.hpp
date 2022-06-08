#pragma once
#include <lcm/lcm-cpp.hpp>
#include "../estimator/d2vinsstate.hpp"
#include <functional>
#include <swarm_msgs/lcm_gen/SlidingWindow_t.hpp>
#include <swarm_msgs/lcm_gen/DistributedSync_t.hpp>
#include <swarm_msgs/lcm_gen/DistributedVinsData_t.hpp>

namespace D2VINS {
class D2Estimator;
struct DistributedVinsData;
class D2VINSNet {
    std::function<void(int, double, std::vector<FrameIdType>)> remote_sld_win_callback;
    D2EstimatorState & state;
    D2Estimator * estimator;
    lcm::LCM lcm;
public:
    std::function<void(DistributedVinsData)> DistributedVinsData_callback;
    std::function<void(int, int, int64_t)> DistributedSync_callback;
    D2VINSNet(D2Estimator * _estimator, std::string lcm_uri);
    void pubSlidingWindow();
    void sendDistributedVinsData(const DistributedVinsData & data);
    void sendSyncSignal(int signal, int64_t token);
    void receiveSyncSignal(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const DistributedSync_t* msg);
    void onSldWinReceived(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const SlidingWindow_t* msg);
    void onDistributedVinsData(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const DistributedVinsData_t * msg);
    int lcmHandle() {
        return lcm.handle();
    }
};
}