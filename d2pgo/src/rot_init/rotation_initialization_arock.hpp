#pragma once
#include "rotation_initialization_base.hpp"
#include <d2common/solver/ARock.hpp>
#include <d2common/d2pgo_types.h>

namespace D2PGO {
template<typename T>
class RotationInitARock: public RotationInitialization<T>, public ARockBase {
protected:
    std::recursive_mutex pgo_data_mutex;
    std::vector<DPGOData> pgo_data;
    std::function<void(const DPGOData &)> broadcastDataCallback;
    bool solve_6d = false;

    virtual SolverReport solveLocalStep() override {
        SolverReport report;
        D2Common::Utility::TicToc tic;
        if (solve_6d) {
            RotationInitialization<T>::solveLinearPose6d();
        } else {
            report.state_changes = RotationInitialization<T>::solveLinearRot();
        }
        report.total_time = tic.toc()/1000;
        report.total_iterations = 1;
        report.succ = true;
        report.message = "";
        return report;
    }

    void setDualStateFactors() {
        for (auto & param_pair : dual_states_remote) {
            for (auto it : param_pair.second) {
                auto state_pointer = it.first;
                auto param_info = all_estimating_params.at(state_pointer);
                if (RotationInitialization<T>::isFixedFrame(param_info.id)) {
                    continue;
                }
                auto & dual_state = it.second;
                Map<const Matrix<double, 3, 3, RowMajor>> M(dual_state.data());
                Swarm::PosePrior prior(param_info.id, M, Matrix3d::Identity()*config.rho_rot_mat);
                RotationInitialization<T>::pose_priors.emplace_back(prior);
            }
        }
    }

    virtual void addFrameId(FrameIdType _frame_id) override {
        RotationInitialization<T>::addFrameId(_frame_id);
        ParamInfo param_info = createFrameRotMat(state, _frame_id);
        addParam(param_info);
    }

    virtual void prepareSolverInIter(bool final_iter) {
        RotationInitialization<T>::pose_priors.clear();
        RotationInitialization<T>::setPriorFactorsbyFixedParam();
    }

    void scanAndCreateDualStates() {
        for (auto loop: RotationInitialization<T>::loops) {
            std::vector<ParamInfo> params_list{createFrameRotMat(state, loop.keyframe_id_a), 
                createFrameRotMat(state, loop.keyframe_id_b)};
            for (auto param_info: params_list) {
                if (isRemoteParam(param_info)) {
                    auto drone_id = solverId(param_info);
                    if (drone_id!=self_id) {
                        if  (!hasDualState(param_info.pointer, drone_id)) {
                            createDualState(param_info, drone_id, true);
                        }
                    }
                }
            }
        }
    }

    void receiveAll() {
        const std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
        for (auto it = pgo_data.begin(); it != pgo_data.end();) {
            auto data = *it;
            processPGOData(data);
            it = pgo_data.erase(it);
        }
    }

    void broadcastData() {
        const std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
        //broadcast the data.
        for (auto it : dual_states_local) {
            DPGOData data;
            data.stamp = ros::Time::now().toSec();
            data.drone_id = self_id;
            data.target_id = it.first;
            data.reference_frame_id = state->getReferenceFrameId();
            data.type = DPGO_ROT_MAT_DUAL;
            printf("[RotInit%d] broadcast data to %d: size %ld\n", self_id, it.first, it.second.size());
            for (auto it2: it.second) {
                auto ptr = it2.first;
                auto & dual_state = it2.second;
                ParamInfo param = all_estimating_params.at(ptr);
                data.frame_poses[param.id] = Swarm::Pose(); // No pose here
                data.frame_duals[param.id] = dual_state;
            }
            if (broadcastDataCallback)
                broadcastDataCallback(data);
        }
    }

    void processPGOData(const DPGOData & data) {
        printf("[ARockPGO@%d]process DPGOData from %d\n", self_id, data.drone_id);
        auto drone_id = data.drone_id;
        for (auto it: data.frame_duals) {
            auto frame_id = it.first;
            auto & dual = it.second;
            if (state->hasFrame(frame_id)) {
                auto * ptr = state->getRotState(frame_id);
                if (all_estimating_params.find(ptr) != all_estimating_params.end()) {
                    if (data.target_id == self_id || !hasDualState(ptr, drone_id)) {
                        updated = true;
                        bool create = false;
                        auto param_info = all_estimating_params.at(ptr);
                        //Then we check it this param has dual.
                        if (!hasDualState(ptr, drone_id)) {
                            //Then we create a new dual state.
                            createDualState(param_info, drone_id, true);
                            create = true;
                        }
                        //Then we update the dual state.
                        dual_states_remote[drone_id][ptr] = dual;
                        if (create)
                            dual_states_local[drone_id][ptr] = dual;
                        // printf("[ARockRotInit@%d]dual remote for frame_id %ld drone_id %d:\n", 
                        //         self_id, frame_id, drone_id);
                        // std::cout << dual.transpose() << std::endl;
                        // printf("[ARockRotInit@%d]dual local: \n ", self_id);
                        // std::cout << dual_states_local[drone_id][ptr].transpose() << std::endl;
                        // VectorXd avg = (dual_states_local[drone_id][ptr] + dual)/2;
                        // printf("[ARockRotInit@%d]avg dual: \n ", self_id);
                        // std::cout << avg.transpose() << std::endl;
                        // printf("[ARockRotInit@%d]state     : \n", 
                        //         self_id);
                        // std::cout << Map<VectorXd>(ptr, 9).transpose() << "\n" << std::endl;
                    }
                }
            }
        }
    }
public:
    RotationInitARock(PGOState * state, RotInitConfig rot_config, 
            ARockSolverConfig arock_config, 
            std::function<void(const DPGOData &)> _broadcastDataCallback): 
        RotationInitialization<T>(state, rot_config), 
        ARockBase(static_cast<D2State*>(state), arock_config),
        broadcastDataCallback(_broadcastDataCallback) {
    }

    SolverReport solve() {
        RotationInitialization<T>::pose_priors.clear();
        for (auto loop : RotationInitialization<T>::loops) {
            auto frame_id_a = loop.keyframe_id_a;
            auto frame_id_b = loop.keyframe_id_b;
            addFrameId(frame_id_a);
            addFrameId(frame_id_b);
        }
        RotationInitialization<T>::updateFrameIdx();
        return solve_arock();
    }

    void reset() {
        RotationInitialization<T>::reset();
    }

    void addLoops(const std::vector<Swarm::LoopEdge> & good_loops) override {
        RotationInitialization<T>::addLoops(good_loops);
        updated = true;
    }

    void inputDPGOData(const DPGOData & data) {
        // printf("[ARockPGO@%d]input DPGOData from %d\n", self_id, data.drone_id);
        std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
        pgo_data.push_back(data);
    }

};

typedef RotationInitARock<double> RotationInitARockd;
typedef RotationInitARock<float> RotationInitARockf;

}