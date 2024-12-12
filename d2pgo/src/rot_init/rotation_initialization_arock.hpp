#pragma once
#include <d2common/d2pgo_types.h>

#include <d2common/solver/ARock.hpp>

#include "rotation_initialization_base.hpp"

namespace D2PGO {
template <typename T>
class RotationInitARock : public RotationInitialization<T>, public ARockBase {
   protected:
    std::recursive_mutex pgo_data_mutex;
    std::vector<DPGOData> pgo_data;
    std::function<void(const DPGOData &)> broadcastDataCallback;

    virtual SolverReport solveLocalStep() override {
        SolverReport report;
        D2Common::Utility::TicToc tic;
        if (solve_6d) {
            report.state_changes = RotationInitialization<T>::solveLinearPose6d();
        } else {
            report.state_changes = RotationInitialization<T>::solveLinearRot();
        }
        report.total_time = tic.toc() / 1000;
        report.total_iterations = 1;
        report.succ = true;
        report.message = "";
        // printf("[RotationInitARock::solveLocalStep%d] local solve time: %.3f ms state changes %f\n", self_id, report.total_time * 1000, report.state_changes);
        return report;
    }

    void setDualStateFactors() {
        for (auto &param_pair : dual_states_remote) {
            for (auto it : param_pair.second) {
                auto state_pointer = it.first;
                auto param_info = all_estimating_params.at(state_pointer);
                if (RotationInitialization<T>::isFixedFrame(param_info.id)) {
                    continue;
                }
                auto &dual_state = it.second;
                if (solve_6d) {
                    Matrix6d inf = Matrix6d::Identity();
                    inf.block<3, 3>(0, 0) *= config.rho_frame_T;
                    inf.block<3, 3>(3, 3) *= config.rho_frame_theta;
                    // printf("[setDualStateFactors%d] remote dual drone %d: frame_id %d delta:", self_id, 
                            // param_pair.first, param_info.id);
                    // std::cout << dual_state.transpose() << std::endl;
                    Swarm::PosePrior prior = Swarm::PosePrior::createFromDelta(
                        param_info.id, dual_state, inf);
                    RotationInitialization<T>::pose_priors.emplace_back(prior);
                } else {
                    Map<const Matrix<double, 3, 3, RowMajor>> M(
                        dual_state.data());
                    Swarm::PosePrior prior(
                        param_info.id, M,
                        Matrix3d::Identity() * config.rho_rot_mat);
                    RotationInitialization<T>::pose_priors.emplace_back(prior);
                }
            }
        }
    }

    virtual void addFrameId(FrameIdType _frame_id) override {
        RotationInitialization<T>::addFrameId(_frame_id);
        if (solve_6d) {
            ParamInfo param_info = createFramePose(state, _frame_id, true);
            addParam(param_info);
        } else {
            ParamInfo param_info = createFrameRotMat(state, _frame_id);
            addParam(param_info);
        }
    }

    virtual void prepareSolverInIter(bool final_iter) {
        RotationInitialization<T>::pose_priors.clear();
        RotationInitialization<T>::setPriorFactorsbyFixedParam();
    }

    void scanAndCreateDualStates() {
        for (auto loop : RotationInitialization<T>::loops) {
            std::vector<ParamInfo> params_list;
            if (solve_6d) {
                params_list = std::vector<ParamInfo>{
                    createFramePose(state, loop.keyframe_id_a, true),
                    createFramePose(state, loop.keyframe_id_b, true)};
            } else {
                params_list = std::vector<ParamInfo>{
                    createFrameRotMat(state, loop.keyframe_id_a),
                    createFrameRotMat(state, loop.keyframe_id_b)};
            }
            for (auto param_info : params_list) {
                if (isRemoteParam(param_info)) {
                    auto drone_id = solverId(param_info);
                    if (drone_id != self_id) {
                        if (!hasDualState(param_info.pointer, drone_id)) {
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
        // broadcast the data.
        for (auto it : dual_states_local) {
            DPGOData data;
            data.stamp = ros::Time::now().toSec();
            data.drone_id = self_id;
            data.target_id = it.first;
            data.reference_frame_id = state->getReferenceFrameId();
            if (solve_6d) {
                data.type = DPGO_DELTA_POSE_DUAL;
            } else {
                data.type = DPGO_ROT_MAT_DUAL;
            }
            for (auto it2 : it.second) {
                auto ptr = it2.first;
                auto &dual_state = it2.second;
                // printf("[broadcastData%d] local dual drone %d: frame_id %d delta:", self_id, 
                //         data.target_id, all_estimating_params.at(ptr).id);
                // std::cout << dual_state.transpose() << std::endl;
                ParamInfo param = all_estimating_params.at(ptr);
                data.frame_poses[param.id] =
                    state->getFramebyId(param.id)->odom.pose();
                data.frame_duals[param.id] = dual_state;
            }
            if (broadcastDataCallback) broadcastDataCallback(data);
        }
    }

    void processPGOData(const DPGOData &data) {
        // printf("[ARockPGO@%d]process DPGOData from %d\n", self_id,
        // data.drone_id);
        auto drone_id = data.drone_id;
        for (auto it : data.frame_duals) {
            auto frame_id = it.first;
            auto &dual = it.second;
            if (state->hasFrame(frame_id)) {
                StatePtr ptr;
                if (solve_6d) {
                    ptr = state->getPerturbState(frame_id);
                } else {
                    ptr = state->getRotState(frame_id);
                }
                if (all_estimating_params.find(ptr) !=
                    all_estimating_params.end()) {
                    if (data.target_id == self_id ||
                        !hasDualState(ptr, drone_id)) {
                        updated = true;
                        bool create = false;
                        auto param_info = all_estimating_params.at(ptr);
                        // Then we check it this param has dual.
                        if (!hasDualState(ptr, drone_id)) {
                            // Then we create a new dual state.
                            createDualState(param_info, drone_id, true);
                            create = true;
                        }
                        // Then we update the dual state.
                        // printf("[processPGOData%d] remote dual drone %d: frame_id %d dual:", self_id, 
                        //         drone_id, param_info.id);
                        // std::cout << dual.transpose() << std::endl;
                        dual_states_remote[drone_id][ptr] = dual;
                        if (create) 
                            dual_states_local[drone_id][ptr] = dual;
                    }
                } else {
                    ROS_WARN("[ARockPGO@%d]process DPGOData from %d, frame_id %ld not found\n", self_id, data.drone_id, frame_id);
                }
            }
        }
    }
public:
    bool solve_6d = false;
    RotationInitARock(PGOState * state, RotInitConfig rot_config, 
            ARockSolverConfig arock_config, 
            std::function<void(const DPGOData &)> _broadcastDataCallback): 
        RotationInitialization<T>(state, rot_config), 
        ARockBase(static_cast<D2State*>(state), arock_config),
        broadcastDataCallback(_broadcastDataCallback) {
            RotationInitialization<T>::is_multi = true;
            config.dual_state_init_to_zero = true;
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

    void addLoops(const std::vector<Swarm::LoopEdge> &good_loops) override {
        RotationInitialization<T>::addLoops(good_loops);
        updated = true;
    }

    void inputDPGOData(const DPGOData &data) {
        // printf("[ARockPGO@%d]input DPGOData from %d\n", self_id,
        // data.drone_id);
        std::lock_guard<std::recursive_mutex> lock(pgo_data_mutex);
        pgo_data.push_back(data);
        for (auto it : data.frame_duals) {
            auto frame_id = it.first;
            if (RotationInitialization<T>::state->hasFrame(frame_id)) {
                auto frame = state->getFramebyId(frame_id);
                if (frame->drone_id == data.drone_id) {
                    auto pose = data.frame_poses.at(frame_id);
                    frame->odom.pose() = pose;
                    RotationInitialization<T>::state->setAttitudeInit(
                        frame_id, pose.att());
                    pose.to_vector(state->getPoseState(frame_id));
                    // printf("Frame %d pose updated from drone %d\n", frame_id, data.drone_id);
                }
            }
        }
    }
};

typedef RotationInitARock<double> RotationInitARockd;
typedef RotationInitARock<float> RotationInitARockf;

}  // namespace D2PGO