#include "ARockPGO.hpp"
#include "d2pgo.h"

namespace D2PGO {

void ARockPGO::inputDPGOData(const DPGOData & data) {
    // printf("[ARockPGO@%d]input DPGOData from %d\n", self_id, data.drone_id);
    const std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
    pgo_data.emplace_back(data);
}

void ARockPGO::processPGOData(const DPGOData & data) {
    // printf("[ARockPGO@%d]process DPGOData from %d\n", self_id, data.drone_id);
    auto drone_id = data.drone_id;
    for (auto it: data.frame_duals) {
        auto frame_id = it.first;
        auto & dual = it.second;
        if (state->hasFrame(frame_id)) {
            auto * ptr = state->getPoseState(frame_id);
            if (all_estimating_params.find(ptr) != all_estimating_params.end()) {
                if (data.target_id == self_id || !hasDualState(ptr, drone_id)) {
                    // if (data.target_id != self_id) {
                    //     printf("[ARockPGO@%d] Discover dual for %ld from %d->%d\n", self_id, frame_id, data.drone_id, data.target_id);
                    // }
                    updated = true;
                    bool create = false;
                    auto param_info = all_estimating_params.at(ptr);
                    //Then we check it this param has dual.
                    if (!hasDualState(ptr, drone_id)) {
                        //Then we create a new dual state.
                        createDualState(param_info, drone_id);
                        create = true;
                    }
                    //Then we update the dual state.
                    if (param_info.type == ParamsType::POSE) {
                        Swarm::Pose pose(dual);
                        pose.to_vector(dual_states_remote[drone_id][ptr].data());
                        if (create)
                            pose.to_vector(dual_states_local[drone_id][ptr].data());
                        // printf("[ARockPGO@%d]dual remote for frame_id %ld drone_id %d: %s\n", 
                        //         self_id, frame_id, drone_id, pose.toStr().c_str());
                        // printf("[ARockPGO@%d]dual local: %s\n", 
                        //         self_id, Swarm::Pose(dual_states_local[drone_id][ptr]).toStr().c_str());
                        // printf("[ARockPGO@%d]state     : %s\n", 
                        //         self_id, Swarm::Pose(ptr, true).toStr().c_str());
                    } else if (param_info.type == ParamsType::POSE_4D) {
                        dual_states_remote[drone_id][ptr] = dual;
                        if (create)
                            dual_states_local[drone_id][ptr] = dual;
                        printf("[ARockPGO@%d]dual remote for frame_id %ld drone_id %d: %s\n", 
                                self_id, frame_id, drone_id, Swarm::Pose(dual).toStr().c_str());
                        printf("[ARockPGO@%d]dual local: %s\n", 
                                self_id, Swarm::Pose(dual_states_local[drone_id][ptr]).toStr().c_str());
                        printf("[ARockPGO@%d]state     : %s\n", 
                                self_id, Swarm::Pose(ptr, true).toStr().c_str());
                    }
                }
            }
        }
    }
}

void ARockPGO::receiveAll() {
    const std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
    //Process and delete data in pgo_data.
    for (auto it = pgo_data.begin(); it != pgo_data.end();) {
        auto data = *it;
        processPGOData(data);
        it = pgo_data.erase(it);
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
            data.frame_duals[param.id] = dual_state;
            data.frame_poses[param.id] = pose;
        }
        pgo->broadcastData(data);
    }
}

void ARockPGO::setStateProperties() {
    pgo->setStateProperties(getProblem());
}

};