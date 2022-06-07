#include "ConsensusSolver.hpp"
#include "ceres/normal_prior.h"
#include "../../factors/consenus_factor.h"

namespace D2VINS {
void ConsensusSolver::addResidual(ResidualInfo*residual_info) {
    for (auto param: residual_info->paramsList(state)) {
        addParam(param);
    }
    residuals.push_back(residual_info);
    problem.AddResidualBlock(residual_info->cost_function, residual_info->loss_function,
            residual_info->paramsPointerList(state));
}

void ConsensusSolver::addParam(const ParamInfo & param_info) {
    if (params.find(param_info.pointer) != params.end()) {
        return;
    }
    params[param_info.pointer] = param_info;
    consenus_params[param_info.pointer] = ConsenusParamState::create(param_info);
    // if (!consenus_params[param_info.pointer].local_only) {
    //     // printf("add param %p to remote_params drone %d\n", param_info.pointer, self_id);
    //     remote_params[param_info.pointer] = std::map<int, VectorXd>();
    //     remote_params[param_info.pointer][self_id] = VectorXd(param_info.size);
    // }
}

ceres::Solver::Summary ConsensusSolver::solve() {
    ceres::Solver::Summary summary;
    for (int i = 0; i < config.max_steps; i++) {
        //If sync mode.
        if (config.is_sync) {
            waitForSync();
            if (i > 0) {
                updateGlobal();
            }
        }
        summary = solveLocalStep();
    }
    return summary;
}

void ConsensusSolver::waitForSync() {
    //Wait for all remote drone to publish result.
}

void ConsensusSolver::updateTilde() {
    for (auto it: params) {
        auto pointer = it.first;
        auto paraminfo = it.second;
        auto & consenus_param = consenus_params[pointer];
        if (consenus_param.local_only) {
            //Add normal prior factor
            //Assmue is a vector.
            Eigen::Map<VectorXd> prior_ref(paraminfo.pointer, paraminfo.size);
            MatrixXd A(paraminfo.size, paraminfo.size);
            A.setIdentity();
            if (paraminfo.type == LANDMARK) {
                A *= rho_landmark;
            } else {
                //Not implement yet
            }
            auto factor = new ceres::NormalPrior(A, prior_ref);
            problem.AddResidualBlock(factor, nullptr, pointer);
        }
        if (IsSE3(paraminfo.type)) {
            //Is SE(3) pose.
            Swarm::Pose pose_global(consenus_param.param_global.data());
            Swarm::Pose pose_local(pointer);
            Swarm::Pose pose_err = Swarm::Pose::DeltaPose(pose_global, pose_local);
            auto & tilde = consenus_param.param_tilde;
            tilde += pose_err.tangentSpace();
            auto factor = new ConsenusPoseFactor(pose_local.pos(), pose_local.att(), 
                tilde.segment<3>(0), tilde.segment<3>(3), rho_T, rho_theta);
            problem.AddResidualBlock(factor, nullptr, pointer);
        } else {
            //Is euclidean.
            VectorXd x_global = consenus_param.param_global;
            Eigen::Map<VectorXd> x_local(pointer, consenus_param.global_size);
            auto & tilde = consenus_param.param_tilde;
            tilde += x_local - x_global;
            MatrixXd A(paraminfo.size, paraminfo.size);
            A.setIdentity();
            auto factor = new ceres::NormalPrior(A, x_global - tilde);
            problem.AddResidualBlock(factor, nullptr, pointer);
        }
    }
}

void ConsensusSolver::updateGlobal() {
    //Assmue all drone's information has been received.
    for (auto it : params) {
        auto pointer = it.first;
        auto paraminfo = it.second;
        if (consenus_params[pointer].local_only) {
            continue;
        }
        if (IsSE3(paraminfo.type)) {
            //Average SE(3) pose.
            std::vector<Swarm::Pose> poses;
            for (auto drone_id: state->availableDrones()) {
                Swarm::Pose pose_(remote_params.at(pointer).at(drone_id).data());
                poses.emplace_back(pose_);
            }
            auto pose_global = Swarm::Pose::averagePose(poses);
            pose_global.to_vector(consenus_params[pointer].param_global.data());
        } else {
            //Average vectors
            VectorXd x_global_sum(paraminfo.eff_size);
            x_global_sum.setZero();
            int drone_count = 0;
            for (auto drone_id: state->availableDrones()) {
                if (remote_params.find(pointer) == remote_params.end()) {
                    printf("\033[0;31mError: remote_params %d of type %d not found in remote_params.\033[0m\n", 
                        paraminfo.id, paraminfo.type);
                    continue;
                }
                if (remote_params.at(pointer).find(drone_id) == remote_params.at(pointer).end()) {
                    printf("\033[0;31mError: remote_params %d type %d of drone %d not found in remote_params.\033[0m\n",
                        paraminfo.id, paraminfo.type, drone_id);
                    continue;
                }
                x_global_sum += remote_params.at(pointer).at(drone_id);
                drone_count += 1;
            }
            if (drone_count == 0) {
                printf("\033[0;31mError: no drone found for %d of type %d.\033[0m\n", paraminfo.id, paraminfo.type);
                continue;
            }
            consenus_params.at(pointer).param_global = x_global_sum / drone_count;
        }
    }
    remote_params.clear();
}

ceres::Solver::Summary ConsensusSolver::solveLocalStep() {
    updateTilde();
    ceres::Solver::Summary summary;
    ceres::Solve(config.ceres_options, &problem, &summary);

    //sync pointers to remote_params.
    for (auto it: consenus_params) {
        auto pointer = it.first;
        auto consenus_param = it.second;
        if (!it.second.local_only) {
            remote_params[pointer] = std::map<int, VectorXd>();
            remote_params[pointer][self_id] = VectorXd(consenus_param.global_size);
            memcpy(remote_params.at(pointer).at(self_id).data(), pointer, consenus_param.global_size);
        }
    }
    return summary;
}

}