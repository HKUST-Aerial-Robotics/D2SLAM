#include "d2vins_net.hpp"

#include <swarm_msgs/swarm_lcm_converter.hpp>

#include "../estimator/d2estimator.hpp"

namespace D2VINS {
D2VINSNet::D2VINSNet(D2Estimator* _estimator, std::string _lcm_uri)
    : lcm(_lcm_uri), estimator(_estimator), state(_estimator->getState()) {
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
  for (unsigned int i = 0; i < state.size(); i++) {
    auto& frame = state.getFrame(i);
    sld_win.frame_ids.push_back(frame.frame_id);
  }
  lcm.publish("SYNC_SLDWIN", &sld_win);
}

void D2VINSNet::sendSyncSignal(int signal, int64_t token) {
  DistributedSync_t sync_signal;
  sync_signal.drone_id = params->self_id;
  sync_signal.sync_signal = signal;
  sync_signal.timestamp = toLCMTime(ros::Time::now());
  sync_signal.solver_token = token;
  lcm.publish("SYNC_SIGNAL", &sync_signal);
}

void D2VINSNet::receiveSyncSignal(const lcm::ReceiveBuffer* rbuf,
                                  const std::string& chan,
                                  const DistributedSync_t* msg) {
  if (msg->drone_id == params->self_id) {
    return;
  }
  DistributedSync_callback(msg->drone_id, msg->sync_signal, msg->solver_token);
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
                                      const DistributedVinsData_t* msg) {
  if (msg->drone_id == params->self_id) {
    return;
  }
  DistributedVinsData_callback(DistributedVinsData(*msg));
}

void D2VINSNet::sendDistributedVinsData(const DistributedVinsData& data) {
  DistributedVinsData_t msg = data.toLCM();
  if (params->print_network_status) {
    printf(
        "[D2VINS] Broadcast VINS Data size %ld with %ld poses %ld extrinsic.\n",
        msg.getEncodedSize(), data.frame_poses.size(), data.extrinsic.size());
  }
  lcm.publish("DISTRIB_VINS_DATA", &msg);
}
}  // namespace D2VINS