#include <d2common/solver/ConsensusSolver.hpp>
#include <ceres/normal_prior.h>
#include <d2common/solver/consenus_factor.h>
#include <d2common/solver/BaseParamResInfo.hpp>

namespace D2Common {
void ConsensusSolver::addResidual(ResidualInfo*residual_info) {
    for (auto param: residual_info->paramsList(state)) {
        addParam(param);
    }
    SolverWrapper::addResidual(residual_info);
}

void ConsensusSolver::reset() {
    SolverWrapper::reset();
    consenus_params.clear();
    all_estimating_params.clear();
    residuals.clear();
    if (config.sync_with_main) {
        remote_params.clear();
    }
}

void ConsensusSolver::addParam(const ParamInfo & param_info) {
    if (all_estimating_params.find(param_info.pointer) != all_estimating_params.end()) {
        return;
    }
    all_estimating_params[param_info.pointer] = param_info;
    consenus_params[param_info.pointer] = ConsenusParamState::create(param_info);
    // printf("add param type %d %d: ", param_info.type, param_info.id);
    // std::cout << consenus_params[param_info.pointer].param_global.transpose() << std::endl;
}

ceres::Solver::Summary ConsensusSolver::solve() {
    ceres::Solver::Summary summary;
    for (int i = 0; i < config.max_steps; i++) {
        //If sync mode.
        broadcastData();
        if (problem != nullptr) {
            delete problem;
        }
        ceres::Problem::Options problem_options;
        if (i != config.max_steps - 1) {
            problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
            problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
        }
        problem = new ceres::Problem(problem_options);
        for (auto residual_info : residuals) {
            problem->AddResidualBlock(residual_info->cost_function, residual_info->loss_function,
                residual_info->paramsPointerList(state));
        }
        if (config.sync_with_main) {
            waitForSync(); 
        } else {
            receiveAll();
        }
        updateGlobal();
        updateTilde();
        setStateProperties();
        summary = solveLocalStep();
        iteration_count++;
    }
    broadcastData();
    return summary;
}

void ConsensusSolver::updateTilde() {
    for (auto it: all_estimating_params) {
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
            problem->AddResidualBlock(factor, nullptr, pointer);
        } else {
            if (IsSE3(paraminfo.type)) {
                //Is SE(3) pose.
                Swarm::Pose pose_global(consenus_param.param_global.data());
                Swarm::Pose pose_local(pointer);
                Swarm::Pose pose_err = Swarm::Pose::DeltaPose(pose_global, pose_local);
                auto & tilde = consenus_param.param_tilde;
                tilde += (1+config.relaxation_alpha) * pose_err.tangentSpace();
                // printf("update tilde %d:\n", paraminfo.id);
                // printf("[updateTilde%d] frame %d pose_local %s pose_global %s tilde :", self_id, 
                //         paraminfo.id, pose_local.toStr().c_str(), pose_global.toStr().c_str());
                // std::cout << "tilde" << tilde.transpose() << std::endl << std::endl;
                auto factor = new ConsenusPoseFactor(pose_global.pos(), pose_global.att(), 
                    tilde.segment<3>(0), tilde.segment<3>(3), rho_T, rho_theta);
                problem->AddResidualBlock(factor, nullptr, pointer);
            } else {
                //Is euclidean.
                printf("[updateTilde] unknow param type %d id %d", paraminfo.type, paraminfo.id);
                VectorXd x_global = consenus_param.param_global;
                Eigen::Map<VectorXd> x_local(pointer, consenus_param.global_size);
                auto & tilde = consenus_param.param_tilde;
                tilde += x_local - x_global;
                MatrixXd A(paraminfo.size, paraminfo.size);
                A.setIdentity();
                auto factor = new ceres::NormalPrior(A, x_global - tilde);
                problem->AddResidualBlock(factor, nullptr, pointer);
            }
        }
    }
}

void ConsensusSolver::updateGlobal() {
    //Assmue all drone's information has been received.
    for (auto it : all_estimating_params) {
        auto pointer = it.first;
        auto paraminfo = it.second;
        if (consenus_params[pointer].local_only) {
            continue;
        }
        if (IsSE3(paraminfo.type)) {
            //Average SE(3) pose.
            std::vector<Swarm::Pose> poses;
            for (auto drone_id: state->availableDrones()) {
                if (remote_params.at(pointer).find(drone_id) == remote_params.at(pointer).end()) {
                    // printf("\033[0;31mError: remote_params %d type %d of drone %d not found in remote_params.\033[0m\n",
                    //     paraminfo.id, paraminfo.type, drone_id);
                    continue;
                }
                Swarm::Pose pose_(remote_params.at(pointer).at(drone_id).data());
                poses.emplace_back(pose_);
            }
            auto pose_global = Swarm::Pose::averagePoses(poses);
            pose_global.to_vector(consenus_params[pointer].param_global.data());
        } else {
            //Average vectors
            VectorXd x_global_sum(paraminfo.eff_size);
            x_global_sum.setZero();
            int drone_count = 0;
            for (auto drone_id: state->availableDrones()) {
                if (remote_params.find(pointer) == remote_params.end()) {
                    // printf("\033[0;31mError: remote_params %d of type %d not found in remote_params.\033[0m\n", 
                    //     paraminfo.id, paraminfo.type);
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
    ceres::Solver::Summary summary;
    ceres::Solve(config.ceres_options, problem, &summary);
    return summary;
}

}