#include <ceres/normal_prior.h>
#include <d2common/solver/consenus_factor.h>
#include <d2common/solver/consenus_factor_4d.h>

#include <d2common/solver/ARock.hpp>

namespace D2Common {

void ARockBase::reset() {
  dual_states_local.clear();
  dual_states_remote.clear();
  all_estimating_params.clear();
}

void ARockBase::addParam(const ParamInfo& param_info) {
  if (all_estimating_params.find(param_info.pointer) !=
      all_estimating_params.end()) {
    return;
  }
  all_estimating_params[param_info.pointer] = param_info;
}

void ARockBase::updateDualStates() {
  for (auto& param_pair : dual_states_local) {
    auto remote_drone_id = param_pair.first;
    auto& duals = param_pair.second;
    for (auto& it : duals) {
      auto state_pointer = it.first;
      auto param_info = all_estimating_params.at(state_pointer);
      auto& dual_state_local = it.second;
      // Now we need to average the remote and the local dual state.
      if (IsSE3(param_info.type)) {
        // Use pose average.
        Swarm::Pose dual_pose_local(dual_state_local);
        Swarm::Pose dual_pose_remote(
            dual_states_remote.at(remote_drone_id).at(state_pointer));
        std::vector<Swarm::Pose> poses{dual_pose_remote, dual_pose_local};
        Swarm::Pose avg_pose = Swarm::Pose::averagePoses(poses);
        Swarm::Pose cur_est_pose = Swarm::Pose(state_pointer);
        Vector6d pose_err =
            Swarm::Pose::DeltaPose(cur_est_pose, avg_pose).tangentSpace();
        Vector6d delta_state = pose_err * config.eta_k;
        // Retraction the delta state to pose
        // printf("[ARockSolver%d] Pose %d delta_state: ", self_id,
        // param_info.id); std::cout << delta_state.transpose() << std::endl;
        Swarm::Pose dual_pose_local_new =
            dual_pose_local * Swarm::Pose::fromTangentSpace(-delta_state);
        dual_pose_local_new.to_vector(dual_state_local.data());
      } else if (IsPose4D(param_info.type)) {
        VectorXd dual_state_remote =
            dual_states_remote.at(remote_drone_id).at(state_pointer);
        VectorXd avg_state = (dual_state_local + dual_state_remote) / 2;
        Map<VectorXd> cur_est_state(CheckGetPtr(state_pointer), param_info.size);
        VectorXd delta = (avg_state - cur_est_state) * config.eta_k;
        dual_state_local = dual_state_local - delta;
        if (dual_state_local(3) > M_PI || dual_state_local(3) < -M_PI) {
          ROS_WARN("Note: [ARockSolver] Dual state %ld has angle %f\n",
                   param_info.id, dual_state_local(3));
          dual_state_local(3) = Utility::NormalizeAngle(dual_state_local(3));
          ROS_WARN("Normed angle: %f", dual_state_local(3));
        }
      } else {
        // printf("[ARockSolver@%d] type %d frame_id %d\n", self_id,
        // param_info.type, param_info.id);
        // Is a vector.
        VectorXd dual_state_remote =
            dual_states_remote.at(remote_drone_id).at(state_pointer);
        // printf("\nFrame %d \n", param_info.id);
        // std::cout << "Dual state remote:\n" << dual_state_remote.transpose()
        // << std::endl; std::cout << "Dual state local :\n" <<
        // dual_state_local.transpose() << std::endl;
        VectorXd avg_state = (dual_state_local + dual_state_remote) / 2;
        // std::cout << "avg_state: \n" << avg_state.transpose() << std::endl;
        Map<VectorXd> cur_est_state(CheckGetPtr(state_pointer), param_info.size);
        VectorXd delta = (avg_state - cur_est_state) * config.eta_k;
        // std::cout << "cur_est_state: \n" << cur_est_state.transpose() <<
        // std::endl; std::cout << "delta: \n" << delta.transpose() <<
        // std::endl;
        dual_state_local -= delta;
      }
    }
  }
}

bool ARockBase::isRemoteParam(const ParamInfo& param_info) {
  if (param_info.type == ParamsType::POSE ||
      param_info.type == ParamsType::POSE_4D ||
      param_info.type == ParamsType::ROTMAT ||
      param_info.type == ParamsType::POSE_PERTURB_6D) {
    auto frame = state->getFramebyId(param_info.id);
    if (frame->drone_id != self_id) {
      return true;
    }
  }
  return false;
}

int ARockBase::solverId(const ParamInfo& param_info) {
  if (param_info.type == ParamsType::POSE ||
      param_info.type == ParamsType::POSE_4D ||
      param_info.type == ParamsType::ROTMAT ||
      param_info.type == ParamsType::POSE_PERTURB_6D) {
    auto frame = state->getFramebyId(param_info.id);
    return frame->drone_id;
  }
  return -1;
}

bool ARockBase::hasDualState(StatePtr param, int drone_id) {
  if (dual_states_remote.find(drone_id) != dual_states_remote.end()) {
    if (dual_states_remote[drone_id].find(param) !=
        dual_states_remote[drone_id].end()) {
      return true;
    }
  }
  return false;
}

void ARockBase::createDualState(const ParamInfo& param_info, int drone_id,
                                bool init_to_zero) {
  if (dual_states_remote.find(drone_id) == dual_states_remote.end()) {
    dual_states_remote[drone_id] = std::map<StatePtr, VectorXd>();
    dual_states_local[drone_id] = std::map<StatePtr, VectorXd>();
  }
  if (init_to_zero) {
    dual_states_remote[drone_id][param_info.pointer] =
        VectorXd::Zero(param_info.size);
    dual_states_local[drone_id][param_info.pointer] =
        VectorXd::Zero(param_info.size);
  } else {
    dual_states_remote[drone_id][param_info.pointer] =
        Map<VectorXd>(CheckGetPtr(param_info.pointer), param_info.size);
    dual_states_local[drone_id][param_info.pointer] =
        Map<VectorXd>(CheckGetPtr(param_info.pointer), param_info.size);
  }
  updated = true;
}

SolverReport ARockBase::solve_arock() {
  // ROS_INFO("ARockBase::solve");
  SolverReport report;
  Utility::TicToc tic;
  int iter_cnt = 0;
  int total_cnt = 0;
  while (iter_cnt < config.max_steps) {
    // If sync mode.
    receiveAll();
    if (!updated) {
      if (config.verbose)
        printf("[ARock@%d] No new data, skip this step: %d total_cnt %d.\n",
               self_id, iter_cnt, total_cnt);
      usleep(config.skip_iteration_usec);
      total_cnt++;
      if (total_cnt > config.max_wait_steps + config.max_steps) {
        if (config.verbose)
          printf("Exit because exceed max_wait_steps: %d\n", total_cnt);
        break;
      } else {
        continue;
      }
    }
    prepareSolverInIter(total_cnt == config.max_steps - 1);
    scanAndCreateDualStates();
    setDualStateFactors();
    auto _report = solveLocalStep();
    updateDualStates();
    broadcastData();
    clearSolver(iter_cnt == config.max_steps - 1);
    report.compose(_report);
    float changes =
        (_report.initial_cost - _report.final_cost) / _report.initial_cost;
    if (iter_cnt == 0) {
      report.initial_cost = _report.initial_cost;
    }
    if (config.verbose)
      printf(
          "[ARock@%d] substeps: %d/%d total_iterations: %d initial_cost: %.2e "
          "final_cost: %.2e changes: %02.2f%% state_changes: %02.2f%%  time: "
          "%.2fms steps: %d\n",
          self_id, iter_cnt, config.max_steps, report.total_iterations,
          report.initial_cost, report.final_cost, changes * 100,
          _report.state_changes, _report.total_time * 1000,
          report.total_iterations);
    iter_cnt++;
    total_cnt++;
  }
  report.total_time = tic.toc() / 1000;
  return report;
}

void ARockSolver::scanAndCreateDualStates() {
  for (auto res : residuals) {
    auto param_infos = res->paramsList(SolverWrapper::state);
    for (auto param_info : param_infos) {
      if (isRemoteParam(param_info)) {
        auto drone_id = solverId(param_info);
        if (drone_id != self_id) {
          if (!hasDualState(param_info.pointer, drone_id)) {
            // createDualState(param_info, drone_id, true);
            createDualState(param_info, drone_id,
                            config.dual_state_init_to_zero);
          }
        }
      }
    }
  }
}

void ARockSolver::reset() {
  SolverWrapper::reset();
  ARockBase::reset();
}

void ARockSolver::addResidual(const std::shared_ptr<ResidualInfo>& residual_info) {
  for (auto param : residual_info->paramsList(SolverWrapper::state)) {
    addParam(param);
  }
  SolverWrapper::addResidual(residual_info);
  updated = true;
}

void ARockSolver::resetResiduals() { residuals.clear(); }

void ARockSolver::prepareSolverInIter(bool final_iter) {
  ceres::Problem::Options problem_options;
  if (!final_iter) {
    problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  }
  problem = new ceres::Problem(problem_options);
  for (auto residual_info : residuals) {
    problem->AddResidualBlock(
        CheckGetPtr(residual_info->cost_function), residual_info->loss_function.get(), // loss_function maybe nullptr
        residual_info->paramsPointerList(SolverWrapper::state));
  }
  setStateProperties();
}

SolverReport ARockSolver::solve() { return ARockBase::solve_arock(); }

void ARockSolver::clearSolver(bool final_substep) {
  if (problem != nullptr) {
    delete problem;
  }
  problem = nullptr;
  // if (!final_substep) {
  //     //We need to clear dual factors manually.
  //     for (auto factor : dual_factors) {
  //         delete factor;
  //     }
  //     dual_factors.clear();
  // }
}

void ARockSolver::setDualStateFactors() {
  for (auto& param_pair : dual_states_remote) {
    for (auto it : param_pair.second) {
      auto state_pointer = it.first;
      auto param_info = all_estimating_params.at(state_pointer);
      auto dual_state = it.second;
      if (IsSE3(param_info.type)) {
        // Is SE(3) pose.
        Swarm::Pose pose_dual(dual_state);
        auto factor = new ConsenusPoseFactor(pose_dual.pos(), pose_dual.att(),
                                             Vector3d::Zero(), Vector3d::Zero(),
                                             rho_T, rho_theta);
        // printf("[ARockSolver] ConsenusPoseFactor param %ld, drone_id %d
        // pose_dual %s pose_cur %s\n",
        //     param_info.id, param_pair.first, pose_dual.toStr().c_str(),
        //     Swarm::Pose(state_pointer).toStr().c_str());
        problem->AddResidualBlock(factor, nullptr, CheckGetPtr(state_pointer));
        dual_factors.push_back(factor);
      } else if (IsPose4D(param_info.type)) {
        Swarm::Pose pose_dual(dual_state);
        // printf("[ARockSolver] ConsenusPoseFactor4D param %ld, drone_id %d
        // pose_dual %s pose_cur %s\n",
        //     param_info.id, param_pair.first, pose_dual.toStr().c_str(),
        //     Swarm::Pose(state_pointer, true).toStr().c_str());
        auto factor =
            ConsenusPoseFactor4D::Create(pose_dual, rho_T, rho_theta, true);
        problem->AddResidualBlock(factor, nullptr, CheckGetPtr(state_pointer));
        dual_factors.push_back(factor);
      } else if (param_info.type == D2Common::POSE_PERTURB_6D) {
        MatrixXd A(param_info.size, param_info.size);
        A.setIdentity();
        A.block<3, 3>(0, 0) *= sqrt(rho_T);
        A.block<3, 3>(3, 3) *= sqrt(rho_theta);
        auto factor = new ceres::NormalPrior(A, dual_state);
        problem->AddResidualBlock(factor, nullptr, CheckGetPtr(state_pointer));
        dual_factors.push_back(factor);
        // if (self_id == 0) {
        //     printf("[ARockSolver] ConsenusPosePerturbFactor param %ld,
        //     drone_id %d A:\n",
        //         param_info.id, param_pair.first);
        //     std::cout << A << std::endl << "dual_state: " <<
        //     dual_state.transpose() << std::endl;
        // }
      } else {
        // Is euclidean.
        MatrixXd A(param_info.size, param_info.size);
        A.setIdentity();
        if (param_info.type == LANDMARK) {
          A *= rho_landmark;
        } else {
          // Not implement yet
        }
        auto factor = new ceres::NormalPrior(A, dual_state);
        problem->AddResidualBlock(factor, nullptr, CheckGetPtr(state_pointer));
        dual_factors.push_back(factor);
      }
    }
  }
}

SolverReport ARockSolver::solveLocalStep() {
  ceres::Solver::Summary summary;
  ceres::Solve(config.ceres_options, problem, &summary);
  updated = false;
  SolverReport report;
  report.total_iterations =
      summary.num_successful_steps + summary.num_unsuccessful_steps;
  report.total_time = summary.total_time_in_seconds;
  report.initial_cost = summary.initial_cost;
  report.final_cost = summary.final_cost;
  report.summary = summary;
  return report;
}

};  // namespace D2Common
