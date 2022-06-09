#include "ConsensusSolver.hpp"
#include "ceres/normal_prior.h"
#include "../../factors/consenus_factor.h"
#include "ResidualInfo.hpp"
#include "../d2vinsstate.hpp"
#include "../d2estimator.hpp"

namespace D2VINS {
void ConsensusSolver::addResidual(ResidualInfo*residual_info) {
    for (auto param: residual_info->paramsList(state)) {
        addParam(param);
    }
    residuals.push_back(residual_info);
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
        broadcastDistributedVinsData();
        if (config.is_sync) {
            problem = new ceres::Problem();
            for (auto residual_info : residuals) {
                problem->AddResidualBlock(residual_info->cost_function, residual_info->loss_function,
                    residual_info->paramsPointerList(state));
            }
            for (auto it: all_estimating_params) {
                if (it.second.type == LANDMARK) {
                    problem->SetParameterLowerBound(it.first, 0, params->min_inv_dep);
                }
            }
            waitForSync(); 
            updateGlobal();
            updateTilde();
            estimator->setStateProperties();
        }
        summary = solveLocalStep();
        iteration_count++;
    }
    return summary;
}

void ConsensusSolver::waitForSync() {
    //Wait for all remote drone to publish result.
    TicToc tic;
    printf("[ConsensusSolver::waitForSync@%d] token %d iteration %d\n", self_id, solver_token, iteration_count - 1);
    std::vector<DistributedVinsData> sync_datas;
    while (tic.toc() < config.timout_wait_sync) {
        //Wait for remote data
        auto ret = receiver->retrive(solver_token, iteration_count);
        sync_datas.insert(sync_datas.end(), ret.begin(), ret.end());
        usleep(100);
        if (ret.size() == state->availableDrones().size() - 1) {
            break;
        }
    }
    for (auto data: sync_datas) {
        updateWithDistributedVinsData(data);
    }
    printf("[ConsensusSolver::waitForSync@%d] receive finsish %ld/%ld time %.1f/%.1fms\n", 
            self_id, sync_datas.size() + 1, state->availableDrones().size(), tic.toc(), config.timout_wait_sync);
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
        }
        if (IsSE3(paraminfo.type)) {
            //Is SE(3) pose.
            Swarm::Pose pose_global(consenus_param.param_global.data());
            Swarm::Pose pose_local(pointer);
            Swarm::Pose pose_err = Swarm::Pose::DeltaPose(pose_global, pose_local);
            auto & tilde = consenus_param.param_tilde;
            tilde += pose_err.tangentSpace();
            // printf("update tilde %d:\n", paraminfo.id);
            printf("[updateTilde%d] frame %d pose_local %s pose_global %s tilde :", self_id, 
                    paraminfo.id, pose_local.toStr().c_str(), pose_global.toStr().c_str());
            std::cout << "tilde" << tilde.transpose() << std::endl << std::endl;
            auto factor = new ConsenusPoseFactor(pose_global.pos(), pose_global.att(), 
                tilde.segment<3>(0), tilde.segment<3>(3), rho_T, rho_theta);
            problem->AddResidualBlock(factor, nullptr, pointer);
        } else {
            //Is euclidean.
            VectorXd x_global = consenus_param.param_global;
            Eigen::Map<VectorXd> x_local(pointer, consenus_param.global_size);
            auto & tilde = consenus_param.param_tilde;
            tilde += x_local - x_global;
            MatrixXd A(paraminfo.size, paraminfo.size);
            A.setIdentity();
            if (paraminfo.type == LANDMARK) {
                A *= rho_landmark;
            }
            auto factor = new ceres::NormalPrior(A, x_global - tilde);
            problem->AddResidualBlock(factor, nullptr, pointer);
        }
    }
}

void ConsensusSolver::updateGlobal() {
    const Guard lock(sync_data_recv_lock);
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
    std::cout << summary.BriefReport() << std::endl;

    return summary;
}

void ConsensusSolver::broadcastDistributedVinsData() {
    //sync pointers to remote_params.
    for (auto it: consenus_params) {
        auto pointer = it.first;
        auto consenus_param = it.second;
        if (!it.second.local_only) {
            remote_params[pointer] = std::map<int, VectorXd>();
            remote_params[pointer][self_id] = VectorXd(consenus_param.global_size);
            memcpy(remote_params.at(pointer).at(self_id).data(), pointer, consenus_param.global_size*sizeof(state_type));
            // printf("set to pose id %d\n", params[pointer].id);
            // std::cout << "remote_params[pointer][self_id]: " << remote_params[pointer][self_id].transpose() << std::endl;
        }
    }

    DistributedVinsData dist_data;
    for (auto it: all_estimating_params) {
        auto pointer = it.first;
        auto param = it.second;
        if (param.type == POSE) {
            dist_data.frame_ids.emplace_back(param.id);
            dist_data.frame_poses.emplace_back(Swarm::Pose(pointer));
        } else if (param.type == EXTRINSIC) {
            dist_data.cam_ids.emplace_back(param.id);
            dist_data.extrinsic.emplace_back(Swarm::Pose(pointer));
        }
    }
    dist_data.stamp = ros::Time::now().toSec();
    dist_data.drone_id = self_id;
    dist_data.solver_token = solver_token;
    dist_data.iteration_count = iteration_count;
    estimator->sendDistributedVinsData(dist_data);
}

void ConsensusSolver::updateWithDistributedVinsData(const DistributedVinsData & dist_data) {
    for (int i = 0; i < dist_data.frame_ids.size(); i++) {
        auto frame_id = dist_data.frame_ids[i];
        if (state->hasFrame(frame_id)) {
            auto pointer = state->getPoseState(frame_id);
            remote_params[pointer][dist_data.drone_id] = VectorXd(POSE_SIZE);
            dist_data.frame_poses[i].to_vector(remote_params[pointer][dist_data.drone_id].data());
            Swarm::Pose local(pointer);
            printf("[updateWithDistributedVinsData%d]pose id %d remote %s local %s\n", self_id, frame_id, 
                dist_data.frame_poses[i].toStr().c_str(),
                local.toStr().c_str());
        }
    }

    for (int i = 0; i < dist_data.cam_ids.size(); i++) {
        auto cam_id = dist_data.cam_ids[i];
        if (state->hasCamera(cam_id)) {
            auto pointer = state->getExtrinsicState(cam_id);
            remote_params[pointer][dist_data.drone_id] = VectorXd(POSE_SIZE);
            dist_data.extrinsic[i].to_vector(remote_params[pointer][dist_data.drone_id].data());
        }
    }

    //Consenus Pwik
    printf("[ConsensusSolver::updateWithDistributedVinsData@%d] of drone %ld: solver token %ld iteration %ld\n",
        self_id, dist_data.drone_id, dist_data.solver_token, dist_data.iteration_count);
}

}