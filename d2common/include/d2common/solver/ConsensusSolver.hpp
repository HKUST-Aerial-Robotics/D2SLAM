#pragma once
#include <ceres/ceres.h>
#include "BaseParamResInfo.hpp"
#include <swarm_msgs/Pose.h>
#include <mutex>
#include "SolverWrapper.hpp"

namespace D2Common {
struct ConsensusSolverConfig {
    int max_steps = 2;
    ceres::Solver::Options ceres_options;
    bool is_sync = true;
    int self_id = 0;
    int main_id = 1;
    double timout_wait_sync = 100;
    double rho_landmark = 0.0;
    double rho_frame_T = 0.1;
    double rho_frame_theta = 0.1;
    double relaxation_alpha = 0.6;
    bool sync_for_averaging = true;
    bool verbose = false;
};

struct ConsenusParamState {
    VectorXd param_global; //global averaged
    VectorXd param_tilde; //tilde
    int global_size = 0;
    int tilde_size = 0;
    ParamsType param_type;
    bool local_only = false;
    static ConsenusParamState create(ParamInfo info) {
        ConsenusParamState state;
        state.param_global = VectorXd(info.size);
        memcpy(state.param_global.data(), info.getPointer(), sizeof(state_type) * info.size);
        state.param_tilde = VectorXd(info.eff_size);
        state.param_tilde.setZero();
        state.global_size = info.size;
        state.tilde_size = info.eff_size;
        state.param_type = info.type;
        if (info.type == LANDMARK || info.type == TD || info.type == SPEED_BIAS) {
            state.local_only = true;
        }
        return state;
    }
};


class ConsensusSolver : public SolverWrapper {
protected:
    ConsensusSolverConfig config;
    std::map<StatePtr, ParamInfo> all_estimating_params;
    std::map<StatePtr, ConsenusParamState> consenus_params;
    std::map<StatePtr, std::map<int, VectorXd>> remote_params;
    std::set<StatePtr> active_params;
    double rho_landmark = 0.1;
    double rho_T = 0.1;
    double rho_theta = 0.1;
    int self_id = 0;
    int solver_token;
    int iteration_count = 0;

    virtual void broadcastData() = 0;
    virtual void receiveAll() = 0;
    virtual void waitForSync() = 0;
    ceres::Solver::Summary solveLocalStep();
    void updateTilde();
    void updateGlobal();
    void addParam(const ParamInfo & param_info);
    void removeDeactivatedParams();
    void syncData();
public:
    ConsensusSolver(D2State * _state, ConsensusSolverConfig _config, int _solver_token): 
        SolverWrapper(_state), config(_config), self_id(config.self_id), solver_token(_solver_token)
    {
        rho_landmark = config.rho_landmark;
        rho_T = config.rho_frame_T;
        rho_theta = config.rho_frame_theta;
    }

    void reset() override;

    virtual void addResidual(const std::shared_ptr<ResidualInfo>& residual_info) override;
    SolverReport solve() override;
    virtual SolverReport solve(std::function<void()> func_set_properties) override { assert(false && "Unused");}
    void setToken(int token) {
        solver_token = token;
    }
    

};
}
