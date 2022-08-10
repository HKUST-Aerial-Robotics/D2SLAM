#pragma once
#include "rotation_initialization_base.hpp"
#include <d2common/solver/ARock.hpp>

namespace D2PGO {
template<typename T>
class RotationInitARock: public RotationInitialization<T>, public ARockBase {
protected:
    virtual SolverReport solveLocalStep() override {
        SolverReport report;
        D2Common::Utility::TicToc tic;
        RotationInitialization<T>::solveLinear();
        report.total_time = tic.toc();
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
                if (param_info.id == RotationInitialization<T>::fixed_frame_id) {
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

    void receiveAll() {
        //TODO
    }

    void broadcastData() {
        //TODO
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
                            createDualState(param_info, drone_id);
                        }
                    }
                }
            }
        }
    }
public:
    RotationInitARock(PGOState * state, RotInitConfig rot_config, ARockSolverConfig arock_config): 
            RotationInitialization<T>(state, rot_config), 
            ARockBase(static_cast<D2State*>(state), arock_config)
    {
    }

    void solve() {
        RotationInitialization<T>::pose_priors.clear();
        solve_arock();
    }

    void reset() {
        RotationInitialization<T>::reset();
    }
};

typedef RotationInitARock<double> RotationInitARockd;
typedef RotationInitARock<float> RotationInitARockf;

}