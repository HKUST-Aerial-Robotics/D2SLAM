#pragma once

#include "SolverWrapper.hpp"

namespace D2Common {
struct ARockSolverConfig {
    int self_id = 0;
    double rho_frame_T = 0.1;
    double rho_frame_theta = 0.1;
    double rho_landmark = 0.1;
    double rho_rot_mat = 0.1;
    double eta_k = 0.9;
    int max_steps = 10;
    int max_wait_steps = 10;
    int skip_iteration_usec = 10000;
    bool verbose = false;
    bool dual_state_init_to_zero = false;
    ceres::Solver::Options ceres_options;
};

class ARockBase {
protected:
    D2State * state;
    bool updated = false;
    ARockSolverConfig config;
    int iteration_count = 0;
    int self_id = 0;
    std::map<int, std::map<StatePtr, VectorXd>> dual_states_local;
    std::map<int, std::map<StatePtr, VectorXd>> dual_states_remote;
    std::map<StatePtr, ParamInfo> all_estimating_params;
    void addParam(const ParamInfo & param_info);
    void updateDualStates();
    bool hasDualState(StatePtr param, int drone_id);
    void createDualState(const ParamInfo & param_info, int drone_id, bool init_to_zero = false);
    virtual bool isRemoteParam(const ParamInfo & param);
    virtual int solverId(const ParamInfo & param);
    
    virtual SolverReport solveLocalStep() = 0;
    virtual void prepareSolverInIter(bool final_iter) = 0;
    virtual SolverReport solve_arock();
    virtual void receiveAll() = 0;
    virtual void broadcastData() = 0;
    virtual void setDualStateFactors() = 0;
    virtual void scanAndCreateDualStates() = 0;
    virtual void clearSolver(bool final_substep) {};
public:
    void reset();
    ARockBase(D2State * _state, ARockSolverConfig _config):
        state(_state), config(_config), self_id(config.self_id) 
    {}
};

class ARockSolver : public SolverWrapper, public ARockBase {
protected:
    double rho_landmark = 0.1;
    double rho_T = 0.1;
    double rho_theta = 0.1;
    virtual SolverReport solveLocalStep() override;
    void setDualStateFactors() override;
    virtual void prepareSolverInIter(bool final_iter) override;
    std::vector<ceres::CostFunction*> dual_factors;
    virtual void clearSolver(bool final_substep) override;
public:
    ARockSolver(D2State * _state, ARockSolverConfig _config):
            SolverWrapper(_state), ARockBase(_state, _config) {
        rho_landmark = config.rho_landmark;
        rho_T = config.rho_frame_T;
        rho_theta = config.rho_frame_theta;
    }
    void reset() override;
    void scanAndCreateDualStates() override;
    virtual void addResidual(ResidualInfo*residual_info) override;
    SolverReport solve() override;
    void resetResiduals();
};
}