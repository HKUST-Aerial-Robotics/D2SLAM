#include <d2common/solver/ARock.hpp>
#include <d2common/solver/consenus_factor.h>
#include <ceres/normal_prior.h>

namespace D2Common {
void ARockSolver::reset() {
    SolverWrapper::reset();
    dual_states_local.clear();
    dual_states_remote.clear();
}

void ARockSolver::addResidual(ResidualInfo*residual_info) {
    for (auto param: residual_info->paramsList(state)) {
        addParam(param);
    }
    SolverWrapper::addResidual(residual_info);
}

void ARockSolver::addParam(const ParamInfo & param_info) {
    if (all_estimating_params.find(param_info.pointer) != all_estimating_params.end()) {
        return;
    }
    all_estimating_params[param_info.pointer] = param_info;
}

ceres::Solver::Summary ARockSolver::solve() {
    ceres::Solver::Summary summary;
    for (int i = 0; i < config.max_steps; i++) {
        //If sync mode.
        if (problem != nullptr) {
            delete problem;
        }
        ceres::Problem::Options problem_options;
        if (i != config.max_steps - 1) {
            problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        }
        problem = new ceres::Problem(problem_options);
        for (auto residual_info : residuals) {
            problem->AddResidualBlock(residual_info->cost_function, residual_info->loss_function,
                residual_info->paramsPointerList(state));
        }
        receiveAll();
        setDualStateFactors();
        setStateProperties();
        solveLocalStep();
        updateDualStates();
        broadcastData();
    }
    return summary;
}

bool ARockSolver::isRemoteParam(const ParamInfo & param_info) {
    if (param_info.type == ParamsType::POSE || param_info.type == ParamsType::POSE_4D) {
        auto frame = state->getFramebyId(param_info.id);
        if (frame.drone_id != self_id) {
            return true;
        }
    }
    return false;
}

int ARockSolver::solverId(const ParamInfo & param_info) {
    if (param_info.type == ParamsType::POSE || param_info.type == ParamsType::POSE_4D) {
        auto frame = state->getFramebyId(param_info.id);
        return frame.drone_id;
    }
    return -1;
}

void ARockSolver::scanAndCreateDualStates() {
    for (auto res: residuals) {
        auto param_infos = res->paramsList(state);
        for (auto param_info: param_infos) {
            if (isRemoteParam(param_info)) {
                auto drone_id = solverId(param_info);
                if  (!hasDualState(param_info.pointer, drone_id)) {
                    createDualState(param_info, drone_id);
                }
            }
        }
    }
}

bool ARockSolver::hasDualState(state_type* param, int drone_id) {
    if (dual_states_remote.find(drone_id) != dual_states_remote.end()) {
        if (dual_states_remote[drone_id].find(param) != dual_states_remote[drone_id].end()) {
            return true;
        }
    }
    return false;
}

void ARockSolver::createDualState(const ParamInfo & param_info, int drone_id) {
    printf("[ARockSolver] Create dual state for param %ld, drone_id %d\n", param_info.id, drone_id);
    if (dual_states_remote.find(drone_id) == dual_states_remote.end()) {
        dual_states_remote[drone_id] = std::map<state_type*, VectorXd>();
        dual_states_local[drone_id] = std::map<state_type*, VectorXd>();
    }
    dual_states_remote[drone_id][param_info.pointer] = Map<VectorXd>(param_info.pointer, param_info.size);
    dual_states_local[drone_id][param_info.pointer] = Map<VectorXd>(param_info.pointer, param_info.size);
}

void ARockSolver::setDualStateFactors() {
    for (auto & param_pair : dual_states_remote) {
        for (auto it : param_pair.second) {
            auto state_pointer = it.first;
            auto param_info = all_estimating_params.at(state_pointer);
            auto dual_state = it.second;
            if (IsSE3(param_info.type)) {
                //Is SE(3) pose.
                Swarm::Pose pose_dual(dual_state);
                auto factor = new ConsenusPoseFactor(pose_dual.pos(), pose_dual.att(), 
                        Vector3d::Zero(), Vector3d::Zero(), rho_T, rho_theta);
                problem->AddResidualBlock(factor, nullptr, state_pointer);
            } else if (IsPose4D(param_info.type)) {
                //Not implemented.
            } else {
                //Is euclidean.
                MatrixXd A(param_info.size, param_info.size);
                A.setIdentity();
                if (param_info.type == LANDMARK) {
                   A *= rho_landmark;
                } else {
                    //Not implement yet
                }
                auto factor = new ceres::NormalPrior(A, dual_state);
                problem->AddResidualBlock(factor, nullptr, state_pointer);
            }
        }
    }
}

void ARockSolver::updateDualStates() {
    for (auto & param_pair : dual_states_local) {
        auto remote_drone_id = param_pair.first;
        auto & duals = param_pair.second;
        for (auto it: duals) {
            auto * state_pointer = it.first;
            auto param_info = all_estimating_params.at(state_pointer);
            auto & dual_state_local = it.second;
            //Now we need to average the remote and the local dual state.
            if (IsSE3(param_info.type)) {
               //Use pose average.
                Swarm::Pose dual_pose_local(dual_state_local);
                Swarm::Pose dual_pose_remote(dual_states_remote.at(remote_drone_id).at(state_pointer));
                std::vector<Swarm::Pose> poses{dual_pose_remote, dual_pose_local};
                Swarm::Pose avg_pose = Swarm::Pose::averagePoses(poses);
                Swarm::Pose cur_est_pose = Swarm::Pose(state_pointer);
                Vector6d pose_err = Swarm::Pose::DeltaPose(cur_est_pose, avg_pose).tangentSpace();
                Vector6d delta_state = pose_err*config.eta_k;
                //Retraction the delta state to pose
                Swarm::Pose dual_pose_local_new = dual_pose_local * 
                    Swarm::Pose::fromTangentSpace(delta_state);
                dual_pose_local_new.to_vector(dual_state_local.data());
            } else if (IsPose4D(param_info.type)) {
                VectorXd dual_state_remote = dual_states_remote.at(remote_drone_id).at(state_pointer);
                VectorXd avg_state = (dual_state_local + dual_state_remote)/2;
                Map<VectorXd> cur_est_state(state_pointer, param_info.size);
                VectorXd delta = (avg_state - cur_est_state)*config.eta_k;
                delta(3) = Utility::NormalizeAngle(delta(3));
                dual_state_local = dual_state_local + delta;
                dual_state_local(3) = Utility::NormalizeAngle(dual_state_local(3));
            } else {
                //Is a vector.
                VectorXd dual_state_remote = dual_states_remote.at(remote_drone_id).at(state_pointer);
                VectorXd avg_state = (dual_state_local + dual_state_remote)/2;
                Map<VectorXd> cur_est_state(state_pointer, param_info.size);
                VectorXd delta = (avg_state - cur_est_state)*config.eta_k;
                dual_state_local += delta;
            }
        }
    }
}

ceres::Solver::Summary ARockSolver::solveLocalStep() {
    ceres::Solver::Summary summary;
    ceres::Solve(config.ceres_options, problem, &summary);
    return summary;
}

};
