#include "d2vins_net.hpp"
#include "../estimator/d2estimator.hpp"
#include <swarm_msgs/swarm_lcm_converter.hpp>

namespace D2VINS {
D2VINSNet::D2VINSNet(D2Estimator * _estimator) 
    :  estimator(_estimator), state(_estimator->getState()) {
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

void D2VINSNet::onSldWinReceived(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const SlidingWindow_t* msg) {
    if (msg->drone_id == params->self_id) {
        return;
    }
    state.updateSldwin(msg->drone_id, msg->frame_ids);
}

}