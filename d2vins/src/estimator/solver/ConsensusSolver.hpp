#pragma once
#include <ceres/ceres.h>
#include "SolverWrapper.hpp"

namespace D2VINS {
struct ConsensusSolverConfig {
    int max_steps = 2;
    ceres::Solver::Options ceres_options;
    bool is_sync = true;
    int self_id = 0;
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
        memcpy(state.param_global.data(), info.pointer, sizeof(state_type) * info.size);
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
    std::vector<ResidualInfo*> residuals;
    std::map<state_type*, ParamInfo> params;
    std::map<state_type*, ConsenusParamState> consenus_params;
    std::map<state_type*, std::map<int, VectorXd>> remote_params;
    double rho_landmark = 0.1;
    double rho_T = 0.1;
    double rho_theta = 0.1;
    int self_id = 0;
public:
    ConsensusSolver(D2EstimatorState * _state, ConsensusSolverConfig _config): 
        SolverWrapper(_state), config(_config),
        self_id(config.self_id)
    {
        printf("ConsensusSolver: self_id: %d\n", self_id);
    }
    virtual void addResidual(ResidualInfo*residual_info) override;
    ceres::Solver::Summary solve() override;
    ceres::Solver::Summary solveLocalStep();
    void addParam(const ParamInfo & param_info);
    void updateTilde();
    void waitForSync();
    void updateGlobal();
};
}
