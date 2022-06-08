#pragma once
#include <ceres/ceres.h>
#include "SolverWrapper.hpp"
#include "ParamInfo.hpp"
#include <swarm_msgs/lcm_gen/DistributedVinsData_t.hpp>
#include <swarm_msgs/Pose.h>
#include <mutex>

namespace D2VINS {
class D2Estimator;
class D2EstimatorState;
typedef std::lock_guard<std::recursive_mutex> Guard;

struct DistributedVinsData {
    double stamp;
    int drone_id;
    std::vector<FrameIdType> frame_ids;
    std::vector<Swarm::Pose> frame_poses;
    std::vector<CamIdType> cam_ids;
    std::vector<Swarm::Pose> extrinsic;
    std::vector<int> remote_drone_ids;
    std::vector<Swarm::Pose> relative_coordinates;
    DistributedVinsData();
    DistributedVinsData(const DistributedVinsData_t & msg);
    DistributedVinsData_t toLCM() const;
};

struct ConsensusSolverConfig {
    int max_steps = 2;
    ceres::Solver::Options ceres_options;
    bool is_sync = true;
    int self_id = 0;
    int main_id = 1;
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
    std::set<int> data_received;
    std::recursive_mutex buf_lock;
public:
    ConsensusSolver(D2EstimatorState * _state, ConsensusSolverConfig _config): 
        SolverWrapper(_state), config(_config),
        self_id(config.self_id)
    {}
    virtual void addResidual(ResidualInfo*residual_info) override;
    ceres::Solver::Summary solve() override;
    ceres::Solver::Summary solveLocalStep();
    void addParam(const ParamInfo & param_info);
    void updateTilde();
    void waitForSync();
    void updateGlobal();
    void onDistributedVinsData(const DistributedVinsData & dist_data);
};
}
