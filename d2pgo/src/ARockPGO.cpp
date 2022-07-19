#include "ARockPGO.hpp"
#include "d2pgo.h"

namespace D2PGO {

void ARockPGO::inputDPGOData(const DPGOData & data) {
    printf("[ARockPGO@%d]input DPGOData from %d\n", self_id, data.drone_id);
    const std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
    pgo_data.emplace_back(data);
}

void ARockPGO::processPGOData(const DPGOData & data) {
    auto drone_id = data.drone_id;
    for (auto it: data.frame_poses) {
        auto frame_id = it.first;
        auto & pose = it.second;
        if (state->hasFrame(frame_id)) {
            auto * ptr = state->getPoseState(frame_id);
            if (all_estimating_params.find(ptr) != all_estimating_params.end()) {
                auto param_info = all_estimating_params.at(ptr);
                //Then we check it this param has dual.
                if (dual_states_remote.find(drone_id) == dual_states_remote.end() || 
                    dual_states_remote[drone_id].find(ptr) ==  dual_states_remote[drone_id].end()) {
                    //Then we create a new dual state.
                    createDualState(param_info, drone_id);
                }
                //Then we update the dual state.
                if (param_info.type == ParamsType::POSE) {
                    dual_states_remote[drone_id][ptr] = VectorXd(POSE_SIZE);
                    pose.to_vector(dual_states_remote[drone_id][ptr].data());
                } else if (param_info.type == ParamsType::POSE_4D) {
                    dual_states_remote[drone_id][ptr] = VectorXd(POSE4D_SIZE);
                    pose.to_vector_xyzyaw(dual_states_remote[drone_id][ptr].data());
                }
            }
        }
    }
}

void ARockPGO::receiveAll() {
    const std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
    for (auto & data : pgo_data) {
        processPGOData(data);
    }
}

void ARockPGO::broadcastData() {
    // printf("ARockPGO::broadcastData\n");
    const std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
    DPGOData data;
    //broadcast the data.
    for (auto it : dual_states_local) {
        data.stamp = ros::Time::now().toSec();
        data.drone_id = self_id;
        data.target_id = it.first;
        data.reference_frame_id = state->getReferenceFrameId();
        // printf("ARockPGO::broadcastData of drone %d\n", data.target_id);
        for (auto it2: it.second) {
            auto ptr = it2.first;
            auto & dual_state = it2.second;
            ParamInfo param = all_estimating_params.at(ptr);
            Swarm::Pose pose;
            if (param.type == ParamsType::POSE) {
                pose = Swarm::Pose(dual_state.data());
            } else if (param.type == ParamsType::POSE_4D) {
                pose = Swarm::Pose(dual_state.data(), true);
            }
            data.frame_poses[param.id] = pose;
        }
        pgo->broadcastData(data);
    }
}

void ARockPGO::setStateProperties() {
    pgo->setStateProperties(getProblem());
}

};