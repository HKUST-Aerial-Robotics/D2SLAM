#pragma once
#include <ceres/ceres.h>
#include "SolverWrapper.hpp"
#include "ParamInfo.hpp"
#include <swarm_msgs/Pose.h>
#include <mutex>
#include "ConsensusSync.hpp"

namespace D2VINS {
class D2Estimator;
class D2EstimatorState;

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
    bool sync_with_main = true;
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
    std::map<state_type*, ParamInfo> all_estimating_params;
    std::map<state_type*, ConsenusParamState> consenus_params;
    std::map<state_type*, std::map<int, VectorXd>> remote_params;
    double rho_landmark = 0.1;
    double rho_T = 0.1;
    double rho_theta = 0.1;
    int self_id = 0;
    D2Estimator * estimator;
    SyncDataReceiver * receiver;
    int solver_token;
    int iteration_count = 0;

    void updateWithDistributedVinsData(const DistributedVinsData & dist_data);
    void broadcastDistributedVinsData();
public:
    ConsensusSolver(D2Estimator * _estimator, D2EstimatorState * _state, SyncDataReceiver * _receiver, 
            ConsensusSolverConfig _config, int _solver_token): 
        estimator(_estimator), SolverWrapper(_state), receiver(_receiver), config(_config), 
        self_id(config.self_id), solver_token(_solver_token)
    {
        rho_landmark = config.rho_landmark;
        rho_T = config.rho_frame_T;
        rho_theta = config.rho_frame_theta;
    }

    void reset() override;

    virtual void addResidual(ResidualInfo*residual_info) override;
    ceres::Solver::Summary solve() override;
    ceres::Solver::Summary solveLocalStep();
    void addParam(const ParamInfo & param_info);
    void updateTilde();
    void waitForSync();
    void receiveAll();
    void updateGlobal();
};
}
