#include "d2vins_net.hpp"
#include "../estimator/d2estimator.hpp"
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include "../estimator/solver/ConsensusSolver.hpp"

namespace D2VINS {
D2VINSNet::D2VINSNet(D2Estimator * _estimator, std::string _lcm_uri): 
        lcm(_lcm_uri), estimator(_estimator), state(_estimator->getState()) {
    lcm.subscribe("DISTRIB_VINS_DATA", &D2VINSNet::onDistributedVinsData, this);
    lcm.subscribe("SYNC_SIGNAL", &D2VINSNet::receiveSyncSignal, this);
}

void D2VINSNet::pubSlidingWindow() {
    if (state.size() == 0) {
        return;
    }
    SlidingWindow_t sld_win;
    sld_win.timestamp = toLCMTime(ros::Time(state.lastFrame().stamp));
    sld_win.sld_win_len = state.size();
    sld_win.drone_id = params->self_id;
    for (int i = 0; i < state.size(); i ++) {
        auto & frame = state.getFrame(i);
        sld_win.frame_ids.push_back(frame.frame_id);
    }
    lcm.publish("SYNC_SLDWIN", &sld_win);
}

void D2VINSNet::sendSyncSignal(int signal) {
    DistributedSync_t sync_signal;
    sync_signal.drone_id = params->self_id;
    sync_signal.sync_signal = signal;
    sync_signal.timestamp = toLCMTime(ros::Time::now());
    lcm.publish("SYNC_SIGNAL", &sync_signal);
}

void D2VINSNet::receiveSyncSignal(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const DistributedSync_t* msg) {
    if (msg->drone_id == params->self_id) {
        return;
    }
    DistributedSync_callback(msg->drone_id, msg->sync_signal);
}

void D2VINSNet::onSldWinReceived(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const SlidingWindow_t* msg) {
    if (msg->drone_id == params->self_id) {
        return;
    }
    state.updateSldwin(msg->drone_id, msg->frame_ids);
}

void D2VINSNet::onDistributedVinsData(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const DistributedVinsData_t * msg) {
    if (msg->drone_id == params->self_id) {
        return;
    }
    DistributedVinsData_callback(DistributedVinsData(*msg));
}

void D2VINSNet::sendDistributedVinsData(const DistributedVinsData & data) {
    DistributedVinsData_t msg = data.toLCM();
    lcm.publish("DISTRIB_VINS_DATA", &msg);
}
}