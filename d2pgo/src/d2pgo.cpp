#include "d2pgo.h"

#include <d2common/solver/angle_manifold.h>
#include <d2common/solver/pose_local_parameterization.h>

#include <d2common/solver/GravityPrior.hpp>
#include <d2common/solver/RelPoseFactor.hpp>

#include "../test/posegraph_g2o.hpp"
#include "ARockPGO.hpp"
#include "rot_init/rotation_initialization.hpp"

namespace D2PGO {

void D2PGO::addFrame(D2BaseFrame frame) {
  const Guard lock(state_lock);
  if (state.hasFrame(frame.frame_id)) {
    return;
  }
  if (config.is_realtime && state.hasDrone(frame.drone_id)) {
    // Use ego motion and current estimation to predict the frame pose
    auto ego_motion = frame.initial_ego_pose;
    auto cur_est_last_frame =
        state.getFrames(frame.drone_id).back()->odom.pose();
    auto ego_motion_last_frame =
        state.getFrames(frame.drone_id).back()->initial_ego_pose;
    auto pose = cur_est_last_frame *
                Swarm::Pose::DeltaPose(ego_motion_last_frame, ego_motion);
    frame.odom.pose() = pose;
  }
  state.addFrame(frame);
  // printf("[D2PGO@%d]add frame %ld ref %d ego_pose %s pose %s from drone
  // %d\n", self_id, frame.frame_id, frame.reference_frame_id,
  //     frame.initial_ego_pose.toStr().c_str(),
  //     frame.odom.pose().toStr().c_str(), frame.drone_id);
  if (ego_motion_trajs.find(frame.drone_id) == ego_motion_trajs.end()) {
    Swarm::DroneTrajectory traj(frame.drone_id, true);
    ego_motion_trajs[frame.drone_id] = traj;
  }
  ego_motion_trajs[frame.drone_id].push(frame.stamp, frame.initial_ego_pose,
                                        frame.frame_id);
  updated = true;
  is_rot_init_convergence = false;
}

void D2PGO::addLoop(const Swarm::LoopEdge& loop_info, bool add_state_by_loop) {
  const Guard lock(state_lock);
  if (loop_info.relative_pose.pos().norm() > config.loop_distance_threshold) {
    SPDLOG_INFO("[D2PGO@{}]loop distance {:.1f}m too large, ignore", self_id,
           loop_info.relative_pose.pos().norm());
    return;
  }
  all_loops.emplace_back(loop_info);
  all_loops.back().id = all_loops.size() - 1;
  // printf("[D2PGO::addLoop@%d] Add edge %ld<->%ld drone %d<->%d hasKF %d %d\n
  // ", self_id, loop_info.keyframe_id_a,
  //     loop_info.keyframe_id_b, loop_info.id_a, loop_info.id_b,
  //     state.hasFrame(loop_info.keyframe_id_a),
  //     state.hasFrame(loop_info.keyframe_id_b));
  if (add_state_by_loop) {
    // This is for debug...
    if (state.hasFrame(loop_info.keyframe_id_a) &&
        !state.hasFrame(loop_info.keyframe_id_b)) {
      // Add frame idb to state
      D2BaseFrame frame_desc;
      frame_desc.drone_id = loop_info.id_b;
      frame_desc.frame_id = loop_info.keyframe_id_b;
      frame_desc.reference_frame_id =
          state.getFramebyId(loop_info.keyframe_id_a)->reference_frame_id;
      // Initialize pose with known pose a and this loop edge
      frame_desc.odom.pose() =
          state.getFramebyId(loop_info.keyframe_id_a)->odom.pose() *
          loop_info.relative_pose;
      addFrame(frame_desc);
      // printf("[D2PGO@%d]add frame %ld pose %s from drone %d\n", self_id,
      // frame_desc.frame_id,
      //     frame_desc.odom.pose().toStr().c_str(), frame_desc.drone_id);
    } else if (!state.hasFrame(loop_info.keyframe_id_a) &&
               state.hasFrame(loop_info.keyframe_id_b)) {
      // Add frame idb to state
      D2BaseFrame frame_desc;
      frame_desc.drone_id = loop_info.id_a;
      frame_desc.frame_id = loop_info.keyframe_id_a;
      frame_desc.reference_frame_id =
          state.getFramebyId(loop_info.keyframe_id_b)->reference_frame_id;
      // Initialize pose with known pose a and this loop edge
      frame_desc.odom.pose() =
          state.getFramebyId(loop_info.keyframe_id_b)->odom.pose() *
          loop_info.relative_pose.inverse();
      addFrame(frame_desc);
      // printf("[D2PGO@%d]add frame %ld pose %s from drone %d\n", self_id,
      // frame_desc.frame_id,
      //     frame_desc.odom.pose().toStr().c_str(), frame_desc.drone_id);
    }
  }
  updated = true;
  is_rot_init_convergence = false;
}

void D2PGO::inputDPGOData(const DPGOData& data) {
  if (config.mode == PGO_MODE_DISTRIBUTED_AROCK) {
    // printf("[D2PGO@%d]input pgo data from drone %d type %d\n", self_id,
    // data.drone_id, data.type);
    if (data.type == DPGODataType::DPGO_POSE_DUAL && solver != nullptr) {
      static_cast<ARockPGO*>(solver)->inputDPGOData(data);
    } else {
      if (data.type == DPGO_DELTA_POSE_DUAL) {
        if (pose6d_init != nullptr) {
          pose6d_init->inputDPGOData(data);
        } else {
          // printf("[D2PGO@%d]input pgo data from drone %d\n", self_id,
          // data.drone_id);
          if (solver != nullptr)
            static_cast<ARockPGO*>(solver)->inputDPGOData(data);
        }
      } else if (rot_init != nullptr) {
        // printf("Input to rot_init\n");
        rot_init->inputDPGOData(data);
      }
    }
  }
}

void D2PGO::inputDPGOsignal(int drone, const std::string& signal) {
  if (signal == "ROT_INIT_FINISH") {
    rot_init_finished_robots.insert(drone);
  }
}

void D2PGO::waitForRotInitFinish() {
  if (rot_init_finished) {
    return;
  }
  sendSignal("ROT_INIT_FINISH");
  rot_init_finished_robots.insert(self_id);
  D2Common::Utility::TicToc timer;
  int count = 0;
  while (rot_init_finished_robots.size() != available_robots.size() &&
         timer.toc() / 1000 < config.rot_init_timeout) {
    usleep(1000);
    if (count % 100 == 0) {
      sendSignal("ROT_INIT_FINISH");
      if (count % 1000 == 0)
        SPDLOG_INFO("[D2PGO@{}]Waiting for rot init finish of other drones, {}/{}",
               self_id, rot_init_finished_robots.size(),
               available_robots.size());
    }
    count++;
  }
  rot_init_finished = true;
  SPDLOG_INFO("[D2PGO@{}]Rot init finish, {}/{} start perturb PGO", self_id,
         rot_init_finished_robots.size(), available_robots.size());
}

bool D2PGO::solve_multi(bool force_solve) {
  const Guard lock(state_lock);
  if ((state.size(self_id) < config.min_solve_size) && !force_solve) {
    // printf("[D2PGO] Not enough frames to solve %d.\n", state.size(self_id));
    return false;
  }
  if (solver == nullptr) {
    solver = new ARockPGO(&state, this, config.arock_config);
  } else {
    static_cast<ARockPGO*>(solver)->resetResiduals();
    // solver = new ARockPGO(&state, this, config.arock_config);
  }
  // used_frames.clear();
  used_loops.clear();
  // Use available loops for outlier rejection.
  std::vector<Swarm::LoopEdge> available_loops;
  for (const Swarm::LoopEdge& loop_info : all_loops) {
    if (state.hasFrame(loop_info.keyframe_id_a) &&
        state.hasFrame(loop_info.keyframe_id_b)) {
      available_loops.emplace_back(loop_info);
    }
  }
  if (config.enable_pcm) {
    D2Common::Utility::TicToc tic;
    auto good_loops =
        rejection.OutlierRejectionLoopEdges(ros::Time::now(), available_loops);
    SPDLOG_INFO("[D2PGO] Pcm takes {:.1f}ms, good {}/{} loops", tic.toc(),
           good_loops.size(), available_loops.size());
    setupLoopFactors(solver, good_loops);
  } else {
    setupLoopFactors(solver, available_loops);
  }
  if (config.enable_ego_motion) {
    setupEgoMotionFactors(solver);
  }
  if (config.debug_save_g2o_only) {
    saveG2O(true);
    used_loops.clear();
    solver->reset();
    return false;
  }
  if (config.enable_rotation_initialization && !isRotInitConvergence()) {
    rotInitial(used_loops);
    if (config.write_g2o) {
      saveG2O();
    }
    // When use pose6d in rot init, we do not solve ceres.
    solve_count++;
    updated = false;
    usleep(50000);  // In this case we sleep 50ms to let the other drones to
                    // initialize rotation
    return true;
  }
  if (config.debug_rot_init_only) {
    return true;
  }
  // Here if enable_rotation_initialization, we need to wait other robots to
  // finish rotation initialization.
  if (config.enable_rotation_initialization) {
    waitForRotInitFinish();
  }
  if (config.enable_gravity_prior) {
    setupGravityPriorFactors(solver);
  }
  SolverReport report;
  if (config.rot_init_config.enable_pose6d_solver) {
    if (pose6d_init != nullptr) {
      pose6d_init->reset();
    } else {
      pose6d_init =
          new RotInit(&state, config.rot_init_config, config.arock_config,
                      config.mode == PGO_MODE_DISTRIBUTED_AROCK,
                      [&](const DPGOData& data) { this->broadcastData(data); });
    }
    pose6d_init->addLoops(used_loops);
    if (isMain()) {
      pose6d_init->setFixedFrameId(state.headId(self_id));
    }
    report = pose6d_init->solve(true);
  } else {
    report = solver->solve();
  }
  if (!config.perturb_mode) {
    state.syncFromState();
  }
  if (postsolve_callback != nullptr) {
    postsolve_callback();
  }
  if (config.write_g2o) {
    saveG2O();
  }
  // std::cout << report.summary.FullReport() << std::endl;
  SPDLOG_INFO(
      "[D2PGO::solve@{}] solve_count {} mode [multi,{}] total frames {} loops "
      "{} opti_time {:.1f}ms iters {} initial cost {:.2e} final cost {:.2e}",
      self_id, solve_count, config.mode, used_frames.size(), used_loops_count,
      report.total_time * 1000, report.total_iterations, report.initial_cost,
      report.final_cost);
  solve_count++;
  updated = false;
  return true;
}

bool D2PGO::solve_single() {
  const Guard lock(state_lock);
  if (state.size(self_id) < config.min_solve_size || !updated) {
    // printf("[D2PGO] Not enough frames to solve %d.\n", state.size(self_id));
    return false;
  }
  used_loops.clear();

  solver = new CeresSolver(&state, config.ceres_options);
  // used_frames.clear();
  // Use available loops for outlier rejection.
  std::vector<Swarm::LoopEdge> available_loops;
  for (const Swarm::LoopEdge& loop_info : all_loops) {
    if (state.hasFrame(loop_info.keyframe_id_a) &&
        state.hasFrame(loop_info.keyframe_id_b)) {
      available_loops.emplace_back(loop_info);
    }
  }
  if (config.enable_pcm) {
    auto good_loops =
        rejection.OutlierRejectionLoopEdges(ros::Time::now(), available_loops);
    setupLoopFactors(solver, good_loops);
  } else {
    setupLoopFactors(solver, available_loops);
  }
  if (config.enable_ego_motion) {
    setupEgoMotionFactors(solver);
  }

  if (config.enable_rotation_initialization && !isRotInitConvergence()) {
    rotInitial(used_loops);
    if (config.write_g2o) {
      saveG2O();
    }
    // Simply return here, we do solve ceres in.
    delete solver;
    solver = nullptr;
    return solve_single();
  }

  if (config.debug_rot_init_only) {
    // When use pose6d in rot init, we do not solve ceres.
    solve_count++;
    updated = false;
    return true;
  }
  if (config.enable_gravity_prior) {
    setupGravityPriorFactors(solver);
  }
  setStateProperties(solver->getProblem());
  auto report = solver->solve();
  if (config.perturb_mode) {
    postPerturbSolve();
  } else {
    state.syncFromState();
  }
  if (postsolve_callback != nullptr) {
    postsolve_callback();
  }
  if (config.write_g2o) {
    saveG2O();
  }
  SPDLOG_INFO(
      "[D2PGO::solve@{}] solve_count {} mode single,{} total frames {} loops "
      "{} opti_time {:.1f}ms iters {} initial cost {:.2e} final cost {:.2e}",
      self_id, solve_count, config.mode, used_frames.size(), used_loops_count,
      report.total_time * 1000, report.total_iterations, report.initial_cost,
      report.final_cost);
  solve_count++;
  updated = false;
  return true;
}

bool D2PGO::isRotInitConvergence() const {
  return is_rot_init_convergence || !config.enable_rotation_initialization;
}

bool D2PGO::isMain() const { return main_id == self_id; }

void D2PGO::rotInitial(const std::vector<Swarm::LoopEdge>& good_loops) {
  if (rot_init == nullptr) {
    rot_init =
        new RotInit(&state, config.rot_init_config, config.arock_config,
                    config.mode == PGO_MODE_DISTRIBUTED_AROCK,
                    [&](const DPGOData& data) { this->broadcastData(data); });
  } else {
    rot_init->reset();
  }
  rot_init->addLoops(good_loops);
  if (isMain()) {
    SPDLOG_INFO("[D2PGO@{}]rotInitial: set first frame fixed", self_id);
    rot_init->setFixedFrameId(state.headId(self_id));
  }
  SolverReport report = rot_init->solve();
  if (config.mode == PGO_MODE_NON_DIST ||
      (report.state_changes < config.rot_init_state_eps && solve_count > 10)) {
    is_rot_init_convergence = true;
    SPDLOG_INFO("[D2PGO@{}]rotInitial: rot init convergence: {:.1f}%\n", self_id,
           report.state_changes * 100);
  } else {
    SPDLOG_INFO(
        "[D2PGO@{}]rotInitial: rot init not convergence: {:.1f}% solve_count {}",
        self_id, report.state_changes * 100, solve_count);
  }
}

void D2PGO::saveG2O(bool only_self) {
  std::vector<std::shared_ptr<D2BaseFrame>> frames;
  for (auto frame_id : used_frames) {
    if (only_self) {
      if (state.getFramebyId(frame_id)->drone_id == self_id) {
        frames.emplace_back(state.getFramebyId(frame_id));
      }
    } else {
      frames.emplace_back(state.getFramebyId(frame_id));
    }
  }
  SPDLOG_INFO("[D2PGO::saveG2O@{}] save {} frames", self_id, frames.size());
  std::string path = config.g2o_output_path + "g2o_drone_" +
                     std::to_string(self_id) + "_iter_" +
                     std::to_string(save_count) + ".g2o";
  SPDLOG_INFO("Save g2o to {}", path);
  write_result_to_g2o(path, frames, used_loops, config.g2o_use_raw_data);
  path = config.g2o_output_path + "frame_timestamp.txt";
  // output the timestamp of the frames.
  std::fstream file;
  file.open(path.c_str(), std::fstream::out);
  std::cout.precision(9);
  for (auto frame : frames) {
    file << frame->frame_id << "  " << std::fixed << frame->stamp << std::endl;
  }
  file.close();
  save_count++;
}

void D2PGO::evalLoop(const Swarm::LoopEdge& loop) {
  auto factor =
      new RelPoseFactor4D(loop.relative_pose, loop.getSqrtInfoMat4D());
  auto kf_a = state.getFramebyId(loop.keyframe_id_a);
  auto kf_b = state.getFramebyId(loop.keyframe_id_b);
  auto pose_ptr_a = state.getPoseState(loop.keyframe_id_a);
  auto pose_ptr_b = state.getPoseState(loop.keyframe_id_b);
  VectorXd residuals(4);
  auto pose_a = kf_a->odom.pose();
  auto pose_b = kf_b->odom.pose();
  (*factor)(CheckGetPtr(pose_ptr_a), CheckGetPtr(pose_ptr_b), residuals.data());
  SPDLOG_INFO("Loop {}->{}, RelPose {}", loop.keyframe_id_a, loop.keyframe_id_b,
         loop.relative_pose.toStr());
  SPDLOG_INFO("RelPose            Est {}",
         Swarm::Pose::DeltaPose(pose_a, pose_b).toStr());
  std::cout << "sqrt_info\n:" << loop.getSqrtInfoMat4D() << std::endl;
  SPDLOG_INFO("PoseA {} PoseB {} residual:", kf_a->odom.pose().toStr(),
         kf_b->odom.pose().toStr());
  std::cout << residuals.transpose() << "\n" << std::endl;
}

void D2PGO::setupLoopFactors(SolverWrapper* solver,
                             const std::vector<Swarm::LoopEdge>& good_loops) {
  used_loops_count = 0;
  // auto loss_function = new ceres::HuberLoss(1.0);
  auto loss_function = nullptr;
  for (auto loop : good_loops) {
    if (state.hasFrame(loop.keyframe_id_a) &&
        state.hasFrame(loop.keyframe_id_b)) {
      std::shared_ptr<ceres::CostFunction> loop_factor = nullptr;
      if (config.pgo_pose_dof == PGO_POSE_4D) {
        loop_factor = RelPoseFactor4D::Create(loop);
        // this->evalLoop(loop);
      } else {
        if (config.pgo_use_autodiff) {
          if (config.perturb_mode && isRotInitConvergence()) {
            auto qa = state.getAttitudeInit(loop.keyframe_id_a);
            auto qb = state.getAttitudeInit(loop.keyframe_id_b);
            loop_factor = RelPoseFactorPerturbAD::Create(loop, qa, qb);
          } else {
            loop_factor = RelPoseFactorAD::Create(loop);
          }
        } else {
          loop_factor = RelPoseFactor::Create(loop);
        }
      }
      auto res_info = RelPoseResInfo::create(
          loop_factor, loss_function, loop.keyframe_id_a, loop.keyframe_id_b,
          config.pgo_pose_dof == PGO_POSE_4D, config.perturb_mode);
      solver->addResidual(res_info);
      used_frames.insert(loop.keyframe_id_a);
      used_frames.insert(loop.keyframe_id_b);
      auto drone_id_a = state.getFramebyId(loop.keyframe_id_a)->drone_id;
      auto ts_a = state.getFramebyId(loop.keyframe_id_a)->stamp;
      auto drone_id_b = state.getFramebyId(loop.keyframe_id_b)->drone_id;
      auto ts_b = state.getFramebyId(loop.keyframe_id_b)->stamp;
      if (used_latest_ts.find(drone_id_a) == used_latest_ts.end() ||
          ts_a > used_latest_ts[drone_id_a]) {
        used_latest_frames[drone_id_a] = loop.keyframe_id_a;
        used_latest_ts[drone_id_a] = ts_a;
      }
      if (used_latest_ts.find(drone_id_b) == used_latest_ts.end() ||
          ts_b > used_latest_ts[drone_id_b]) {
        used_latest_frames[drone_id_b] = loop.keyframe_id_b;
        used_latest_ts[drone_id_b] = ts_b;
      }
      used_loops_count++;
      used_loops.emplace_back(loop);
      // if (loop.id_a != loop.id_b)
      //     printf("[D2PGO::setupLoopFactors@%d] add loop %ld->%ld pose: %s\n",
      //         self_id, loop.keyframe_id_a, loop.keyframe_id_b,
      //         loop.relative_pose.toStr().c_str());
    }
  }
}

void D2PGO::setupEgoMotionFactors(SolverWrapper* solver, int drone_id) {
  auto frames = state.getFrames(drone_id);
  auto traj = state.getEgomotionTraj(drone_id);
  for (unsigned int i = 0; i < frames.size() - 1; i++) {
    auto frame_a = frames[i];
    auto frame_b = frames[i + 1];
    Swarm::Pose rel_pose;
    if (config.pgo_pose_dof == PGO_POSE_4D) {
      rel_pose = Swarm::Pose::DeltaPose(frame_a->initial_ego_pose,
                                        frame_b->initial_ego_pose, true);
    } else {
      rel_pose = Swarm::Pose::DeltaPose(frame_a->initial_ego_pose,
                                        frame_b->initial_ego_pose);
    }
    double len = rel_pose.pos().norm();
    if (len < config.min_cov_len) {
      len = config.min_cov_len;
    }
    Eigen::Matrix6d cov = Eigen::Matrix6d::Zero();
    cov.block<3, 3>(0, 0) =
        Matrix3d::Identity() * config.pos_covariance_per_meter * len +
        0.5 * Matrix3d::Identity() * config.yaw_covariance_per_meter * len *
            len;
    cov.block<3, 3>(3, 3) =
        Matrix3d::Identity() * config.yaw_covariance_per_meter * len;
    Matrix6d sqrt_info = cov.inverse().cwiseAbs().cwiseSqrt();
    Swarm::LoopEdge loop(frame_a->frame_id, frame_b->frame_id, rel_pose,
                         sqrt_info);
    if (config.pgo_pose_dof == PGO_POSE_4D) {
      auto factor = RelPoseFactor4D::Create(loop);
      auto res_info = RelPoseResInfo::create(factor, nullptr, frame_a->frame_id,
                                             frame_b->frame_id, true);
      solver->addResidual(res_info);
    } else if (config.pgo_pose_dof == PGO_POSE_6D) {
      std::shared_ptr<ceres::CostFunction> factor = nullptr;
      if (config.pgo_use_autodiff) {
        if (config.perturb_mode && isRotInitConvergence()) {
          auto qa = state.getAttitudeInit(frame_a->frame_id);
          auto qb = state.getAttitudeInit(frame_b->frame_id);
          factor = RelPoseFactorPerturbAD::Create(loop, qa, qb);
        } else {
          factor = RelPoseFactorAD::Create(loop);
        }
      } else {
        factor = RelPoseFactor::Create(loop);
      }
      auto res_info =
          RelPoseResInfo::create(factor, nullptr, frame_a->frame_id,
                                 frame_b->frame_id, false, config.perturb_mode);
      solver->addResidual(res_info);
    }
    used_frames.insert(frame_a->frame_id);
    used_frames.insert(frame_b->frame_id);
    used_loops.emplace_back(loop);
    if (used_latest_ts.find(frame_b->frame_id) == used_latest_ts.end() ||
        frame_b->stamp > used_latest_ts[frame_b->drone_id]) {
      used_latest_frames[frame_b->drone_id] = frame_b->frame_id;
      used_latest_ts[frame_b->drone_id] = frame_b->stamp;
    }
  }
}

void D2PGO::setupGravityPriorFactors(SolverWrapper* solver) {
  if (config.pgo_pose_dof == PGO_POSE_4D) {
    return;
  }
  SPDLOG_INFO("[D2PGO::setupGravityPriorFactors] {} frames", used_frames.size());

  Eigen::Matrix3d gravity_sqrt_info =
      config.rot_init_config.gravity_sqrt_info * Matrix3d::Identity();
  for (auto& frame : used_frames) {
    auto frame_ptr = state.getFramebyId(frame);
    auto ego_motion = frame_ptr->initial_ego_pose;
    std::shared_ptr<ceres::CostFunction> factor = nullptr;
    if (config.perturb_mode && isRotInitConvergence()) {
      auto qa = state.getAttitudeInit(frame);
      factor = GravityPriorPerturbAD::Create(ego_motion, gravity_sqrt_info, qa);
    } else {
      // Not implemented
      SPDLOG_ERROR(
          "[D2PGO::setupGravityPriorFactors] non perturb_mode not implemented");
    }
    auto res_info = GravityPriorResInfo::create(factor, nullptr, frame,
                                                config.perturb_mode);
    solver->addResidual(res_info);
    if (used_latest_ts.find(frame) == used_latest_ts.end() ||
        frame_ptr->stamp > used_latest_ts[frame_ptr->drone_id]) {
      used_latest_frames[frame_ptr->drone_id] = frame;
      used_latest_ts[frame_ptr->drone_id] = frame_ptr->stamp;
    }
  }
}

void D2PGO::setupEgoMotionFactors(SolverWrapper* solver) {
  if (config.mode == PGO_MODE_NON_DIST) {
    for (auto drone_id : state.availableDrones()) {
      setupEgoMotionFactors(solver, drone_id);
    }
  } else if (config.mode >= PGO_MODE_DISTRIBUTED_AROCK) {
    setupEgoMotionFactors(solver, self_id);
  }
}

void D2PGO::setStateProperties(ceres::Problem& problem) {
  ceres::Manifold* manifold;
  ceres::LocalParameterization* local_parameterization;
  if (config.pgo_pose_dof == PGO_POSE_4D) {
    manifold = PosAngleManifold::Create();
  } else {
    if (!config.perturb_mode) {
      if (config.pgo_use_autodiff) {
        ceres::EigenQuaternionManifold quat_manifold;
        ceres::EuclideanManifold<3> euc_manifold;
        manifold = new ceres::ProductManifold<ceres::EuclideanManifold<3>,
                                              ceres::EigenQuaternionManifold>(
            euc_manifold, quat_manifold);
      } else {
        local_parameterization = new PoseLocalParameterization;
      }
    }
  }
  if (!config.perturb_mode) {
    for (auto frame_id : used_frames) {
      auto pointer = state.getPoseState(frame_id);
      if (!problem.HasParameterBlock(CheckGetPtr(pointer))) {
        continue;
      }
      if (config.pgo_pose_dof == PGO_POSE_4D || config.pgo_use_autodiff) {
        problem.SetManifold(CheckGetPtr(pointer), manifold);
      } else {
        problem.SetParameterization(CheckGetPtr(pointer), local_parameterization);
      }
    }
  }
  if (config.mode == PGO_MODE_NON_DIST ||
      config.mode >= PGO_MODE_DISTRIBUTED_AROCK && self_id == main_id) {
    auto frame_id = state.headId(self_id);
    if (config.perturb_mode) {
      auto pointer = state.getPerturbState(frame_id);
      problem.SetParameterBlockConstant(CheckGetPtr(pointer));
      // printf("[D2PGO::setStateProperties@%d] set perturb state %ld to
      // constant\n", self_id, frame_id);
    } else {
      auto pointer = state.getPoseState(frame_id);
      problem.SetParameterBlockConstant(CheckGetPtr(pointer));
    }
  }
}

void D2PGO::postPerturbSolve() {
  for (auto frame_id : used_frames) {
    auto pointer = state.getPerturbState(frame_id);
    Map<Vector3d> pos(CheckGetPtr(pointer));
    Map<Vector3d> perturb_theta(CheckGetPtr(pointer) + 3);
    Quaterniond q_perturb = Utility::quatfromRotationVector(perturb_theta);
    Swarm::Pose optimized_pose(pos,
                               state.getAttitudeInit(frame_id) * q_perturb);
    state.getFramebyId(frame_id)->odom.pose() = optimized_pose;
    // perturb_theta.setZero();
  }
}

std::map<int, Swarm::DroneTrajectory> D2PGO::getOptimizedTrajs() {
  const Guard lock(state_lock);
  std::map<int, Swarm::DroneTrajectory> trajs;
  for (auto drone_id : state.availableDrones()) {
    trajs[drone_id] = Swarm::DroneTrajectory(drone_id, false);
    for (auto frame : state.getFrames(drone_id)) {
      if (used_frames.count(frame->frame_id) == 0) {
        continue;
      }
      auto pose = frame->odom.pose();
      if (config.pgo_pose_dof == PGO_POSE_4D) {
        // Then we need to combine the roll pitch from ego motion
        Swarm::Pose ego_pose = frame->initial_ego_pose;
        auto delta_att = ego_pose.att_yaw_only().inverse() * ego_pose.att();
        pose.att() = pose.att() * delta_att;
      }
      if (config.perturb_mode) {
        auto pointer = state.getPerturbState(frame->frame_id);
        Map<Vector3d> pos(CheckGetPtr(pointer));
        Map<Vector3d> perturb_theta(CheckGetPtr(pointer) + 3);
        Quaterniond q_perturb = Utility::quatfromRotationVector(perturb_theta);
        pose = Swarm::Pose(pos,
                           state.getAttitudeInit(frame->frame_id) * q_perturb);
      }
      trajs[drone_id].push(frame->stamp, pose, frame->frame_id);
    }
    if (trajs[drone_id].trajectory_size() == 0) {
      trajs.erase(drone_id);
    }
  }
  return trajs;
}

std::map<int, Swarm::Odometry> D2PGO::getPredictedOdoms() const {
  const Guard lock(state_lock);
  std::map<int, Swarm::Odometry> ret;
  for (auto drone_id : state.availableDrones()) {
    if (!state.hasEgomotionTraj(drone_id)) {
      SPDLOG_INFO("[D2PGO::getPredictedOdoms] no egomotion traj for drone {}",
             drone_id);
      continue;
    }
    const Swarm::DroneTrajectory& ego_motion_traj =
        state.getEgomotionTraj(drone_id);
    if (ego_motion_traj.trajectory_size() > 0 &&
        used_latest_frames.find(drone_id) != used_latest_frames.end()) {
      auto latest_ego_pose = ego_motion_traj.get_latest_pose();
      auto latest_ts = ego_motion_traj.get_latest_stamp();
      auto last_est_frame_id = used_latest_frames.at(drone_id);
      auto last_est_frame = state.getFramebyId(last_est_frame_id);
      auto last_est_ego = last_est_frame->initial_ego_pose;
      auto predicted_pose = last_est_frame->odom.pose() *
                            last_est_ego.inverse() * latest_ego_pose;
      ret[drone_id] = Swarm::Odometry(latest_ts, predicted_pose);
    }
  }
  return ret;
}

void D2PGO::broadcastData(const DPGOData& data) { bd_data_callback(data); }

void D2PGO::sendSignal(const std::string& signal) {
  bd_signal_callback(signal);
}

std::vector<D2BaseFrame*> D2PGO::getAllLocalFrames() {
  const Guard lock(state_lock);
  return state.getFrames(self_id);
}

}  // namespace D2PGO