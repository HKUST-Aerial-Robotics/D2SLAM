#include "d2estimator.hpp"

#include <d2common/solver/pose_local_parameterization.h>
#include <d2frontend/pnp_utils.h>

#include <d2common/utils.hpp>

#include "../factors/depth_factor.h"
#include "../factors/imu_factor.h"
#include "../factors/prior_factor.h"
#include "../factors/projectionOneFrameTwoCamFactor.h"
#include "../factors/projectionTwoFrameOneCamDepthFactor.h"
#include "../factors/projectionTwoFrameOneCamFactor.h"
#include "../factors/projectionTwoFrameTwoCamFactor.h"
#include "../network/d2vins_net.hpp"
#include "marginalization/marginalization.hpp"
#include "solver/ConsensusSync.hpp"
#include "solver/VINSConsenusSolver.hpp"
#include "spdlog/spdlog.h"
#include "unistd.h"

namespace D2VINS {

D2Estimator::D2Estimator(int drone_id) : self_id(drone_id), state(drone_id) {
  sync_data_receiver = new SyncDataReceiver;
}

void D2Estimator::init(ros::NodeHandle &nh, D2VINSNet *net) {
  state.init(params->camera_extrinsics, params->td_initial);
  visual.init(nh, this);
  SPDLOG_INFO("init done estimator on drone {}", self_id);
  for (auto cam_id : state.getAvailableCameraIds()) {
    Swarm::Pose ext = state.getExtrinsic(cam_id);
    SPDLOG_INFO("extrinsic {}: {}", cam_id, ext.toStr());
  }
  vinsnet = net;
  if (vinsnet != nullptr)
  {
    vinsnet->DistributedVinsData_callback = [&](DistributedVinsData msg) {
      onDistributedVinsData(msg);
    };
    vinsnet->DistributedSync_callback = [&](int drone_id, int signal,
                                            int64_t token) {
      onSyncSignal(drone_id, signal, token);
    };
  }

  imu_bufs[self_id] = IMUBuffer();
  if (params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS) {
    solver = new D2VINSConsensusSolver(this, &state, sync_data_receiver,
                                       *params->consensus_config, solve_token);
  } else {
    solver = new CeresSolver(&state, params->ceres_options);
  }
}

void D2Estimator::inputImu(IMUData data) {
  std::lock_guard<std::recursive_mutex> lock(imu_prop_lock);
  IMUData last = data;
  if (imu_bufs[self_id].size() > 0) {
    last = imu_bufs[self_id].buf.back();
  }
  imu_bufs[self_id].add(data);
  if (!initFirstPoseFlag || !isInitialized()) {
    return;
  }
  // Propagation current with last Bias.
  Eigen::Vector3d Ba = state.getBa();
  Eigen::Vector3d Bg = state.getBg();
  data.propagation(last_prop_odom[params->self_id], Ba, Bg, last);
  visual.pubIMUProp(last_prop_odom[params->self_id]);
}

bool D2Estimator::tryinitFirstPose(VisualImageDescArray &frame) {
  auto ret = imu_bufs[self_id].periodIMU(
      -1, frame.stamp + state.getTd(frame.drone_id));
  auto _imubuf = ret.first;
  if (_imubuf.size() < params->init_imu_num) {
    SPDLOG_INFO("not enough imu data {}/{} for init", _imubuf.size(),
                imu_bufs[self_id].size());
    return false;
  }
  auto mean_acc = _imubuf.mean_acc();
  auto q0 = Utility::g2R(mean_acc);
  auto last_odom =
      Swarm::Odometry(frame.stamp, Swarm::Pose(q0, Vector3d::Zero()));
  auto acc_bias = _imubuf.mean_acc() - q0.inverse() * IMUData::Gravity;
  if (acc_bias.norm() > params->init_acc_bias_threshold) {
    imu_bufs[self_id].clear();
    SPDLOG_WARN("Robot not steady: acc bias too large {:.2f} > {:.2f}, init failed. clear buf and wait...", acc_bias.norm(),
                params->init_acc_bias_threshold);
    return false;
  }
  // Easily use the average value as gyrobias now
  // Also the ba with average acc - g
  VINSFrame first_frame(frame, mean_acc - q0.inverse() * IMUData::Gravity,
                        _imubuf.mean_gyro());
  first_frame.is_keyframe = true;
  first_frame.odom = last_odom;
  first_frame.imu_buf_index = ret.second;
  first_frame.reference_frame_id = state.getReferenceFrameId();
  state.addFrame(frame, first_frame);

  SPDLOG_INFO("===============Initialization=============");
  SPDLOG_INFO("Initial firstPose {}", frame.frame_id);
  SPDLOG_INFO("Init pose with IMU: {}", last_odom.toStr());
  SPDLOG_INFO("Mean acc    {:.6f} {:.6f} {:.6f}", mean_acc.x(), mean_acc.y(),
              mean_acc.z());
  SPDLOG_INFO("Gyro bias:  {:.6f} {:.6f} {:.6f}", first_frame.Bg.x(),
              first_frame.Bg.y(), first_frame.Bg.z());
  SPDLOG_INFO("Acc  bias:  {:.6f} {:.6f} {:.6f}\n", first_frame.Ba.x(),
              first_frame.Ba.y(), first_frame.Ba.z());
  SPDLOG_INFO("==========================================");

  frame.reference_frame_id = state.getReferenceFrameId();
  frame.pose_drone = first_frame.odom.pose();
  frame.Ba = first_frame.Ba;
  frame.Bg = first_frame.Bg;
  frame.setTd(state.getTd(frame.drone_id));
  return true;
}

std::pair<bool, Swarm::Pose> D2Estimator::initialFramePnP(
    const VisualImageDescArray &frame, const Swarm::Pose &initial_pose) {
  // Only use first image for initialization.
  std::vector<Vector3d> lm_positions_a, lm_3d_norm_b;
  std::vector<Swarm::Pose> cam_extrinsics;
  std::vector<int> camera_indices, inliers;
  // Compute pose with computePosePnPnonCentral
  int i = 0;
  for (auto image : frame.images) {
    cam_extrinsics.push_back(image.extrinsic);
    for (auto &lm : image.landmarks) {
      auto &lm_id = lm.landmark_id;
      if (state.hasLandmark(lm_id)) {
        auto &est_lm = state.getLandmarkbyId(lm_id);
        if (est_lm.flag >= LandmarkFlag::INITIALIZED) {
          lm_positions_a.push_back(est_lm.position);
          lm_3d_norm_b.push_back(lm.pt3d_norm);
          camera_indices.push_back(lm.camera_index);
        }
      }
    }
  }
  D2Common::Utility::TicToc tic;
  auto pose_imu = D2FrontEnd::computePosePnPnonCentral(
      lm_positions_a, lm_3d_norm_b, cam_extrinsics, camera_indices, inliers);
  bool success = inliers.size() > params->pnp_min_inliers;
  if (frame.drone_id != self_id) {
    SPDLOG_INFO(
        "D{} PnP succ {} frame {}@{} final {} inliers {} points {} "
        "time: {:.2f}ms",
        self_id, success, frame.frame_id, frame.drone_id, pose_imu.toStr(),
        inliers.size(), lm_positions_a.size(), tic.toc());
  }
  return std::make_pair(success, pose_imu);
}

VINSFrame *D2Estimator::addFrame(VisualImageDescArray &_frame) {
  // First we init corresponding pose for with IMU
  auto &last_frame = state.lastFrame();
  auto motion_predict = getMotionPredict(
      _frame.stamp);  // Redo motion predict for get latest initial pose
  VINSFrame frame(_frame, motion_predict.second, last_frame);
  if (params->init_method == D2VINSConfig::INIT_POSE_IMU) {
    frame.odom = motion_predict.first;
  } else {
    auto odom_imu = motion_predict.first;
    auto pnp_init = initialFramePnP(_frame, last_frame.odom.pose());
    if (!pnp_init.first) {
      // Use IMU
      SPDLOG_WARN("Initialization failed, use IMU instead.");
    } else {
      odom_imu.pose() = pnp_init.second;
    }
    frame.odom = odom_imu;
  }
  frame.odom.stamp = _frame.stamp;
  frame.reference_frame_id = state.getReferenceFrameId();

  auto frame_ret = state.addFrame(_frame, frame);
  // Clear old frames after add
  if (params->estimation_mode != D2Common::DISTRIBUTED_CAMERA_CONSENUS) {
    margined_landmarks = state.clearUselessFrames(
        isInitialized());  // Only marginlization when solved
  }
  _frame.setTd(state.getTd(_frame.drone_id));
  // Assign IMU and initialization to VisualImageDescArray for broadcasting.
  _frame.imu_buf = motion_predict.second.first;
  _frame.pose_drone = frame.odom.pose();
  _frame.Ba = frame.Ba;
  _frame.Bg = frame.Bg;
  _frame.reference_frame_id = frame.reference_frame_id;

  spdlog::debug("Initialize VINSFrame with {}: {}", params->init_method,
                frame.toStr().c_str());
  return frame_ret;
}

void D2Estimator::addRemoteImuBuf(int drone_id, const IMUBuffer &imu_) {
  if (imu_bufs.find(drone_id) == imu_bufs.end()) {
    imu_bufs[drone_id] = imu_;
    SPDLOG_INFO("Assign imu buf to drone {} cur_size {}", drone_id,
                imu_bufs[drone_id].size());
  } else {
    auto &_imu_buf = imu_bufs.at(drone_id);
    auto t_last = _imu_buf.t_last;
    bool add_first = true;
    for (size_t i = 0; i < imu_.size(); i++) {
      if (imu_[i].t > t_last) {
        if (add_first) {
          if ((imu_[i].t - t_last) > params->max_imu_time_err) {
            SPDLOG_INFO("Add remote imu buffer {}: dt {:.2f}ms", drone_id,
                        (imu_[i].t - t_last) * 1000);
          }
          add_first = false;
        }
        _imu_buf.add(imu_[i]);
      }
    }
  }
}

VINSFrame *D2Estimator::addFrameRemote(const VisualImageDescArray &_frame) {
  if (params->estimation_mode == D2Common::SOLVE_ALL_MODE ||
      params->estimation_mode == D2Common::SERVER_MODE) {
    addRemoteImuBuf(_frame.drone_id, _frame.imu_buf);
  }
  int r_drone_id = _frame.drone_id;
  VINSFrame vinsframe;
  auto _imu = _frame.imu_buf;
  if (state.size(r_drone_id) > 0) {
    auto last_frame = state.lastFrame(r_drone_id);
    if (params->estimation_mode == D2Common::SOLVE_ALL_MODE ||
        params->estimation_mode == D2Common::SERVER_MODE) {
      auto &imu_buf = imu_bufs.at(_frame.drone_id);
      auto ret =
          imu_buf.periodIMU(last_frame.imu_buf_index, _frame.stamp + state.getTd());
      auto _imu = ret.first;
      if (fabs(_imu.size() / (_frame.stamp - last_frame.stamp) -
               params->IMU_FREQ) > 15) {
        SPDLOG_WARN(
            "Remote IMU error freq: {:.3f} start_t "
            "{:.3f}/{:.3f} end_t {:.3f}/{:.3f}",
            _imu.size() / (_frame.stamp - last_frame.stamp),
            last_frame.stamp + state.getTd(), _imu[0].t, _frame.stamp + state.getTd(),
            _imu[_imu.size() - 1].t);
      }
      vinsframe = VINSFrame(_frame, ret, last_frame);
    } else {
      vinsframe = VINSFrame(_frame, _frame.Ba, _frame.Bg);
    }
    if (_frame.reference_frame_id != state.getReferenceFrameId()) {
      auto ego_last = last_frame.initial_ego_pose;
      auto pose_local_cur = _frame.pose_drone;
      auto pred_cur_pose =
          last_frame.odom.pose() * ego_last.inverse() * pose_local_cur;
      vinsframe.odom.pose() = pred_cur_pose;
      spdlog::debug("Initial remoteframe {}@{} with ego-motion: {}",
                    _frame.frame_id, r_drone_id, pred_cur_pose.toStr());
    }
  } else {
    // Need to init the first frame.
    vinsframe = VINSFrame(_frame, _frame.Ba, _frame.Bg);
    auto pnp_init = initialFramePnP(_frame, Swarm::Pose::Identity());
    if (!pnp_init.first) {
      // Use IMU
      spdlog::debug("Initialization failed for remote {}@{}. will not add",
                    _frame.frame_id, _frame.drone_id);
      return nullptr;
    } else {
      spdlog::debug("Initial first remoteframe@{} with PnP: {}", r_drone_id,
                    pnp_init.second.toStr());
      if (_frame.reference_frame_id < state.getReferenceFrameId() &&
          params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS) {
        // In this case, we merge the current map to the remote.
        auto P_w_ki = _frame.pose_drone * pnp_init.second.inverse();
        P_w_ki.set_yaw_only();
        state.moveAllPoses(_frame.reference_frame_id, P_w_ki);
        SPDLOG_INFO("Merge map to reference frame {}@{} RP: {}",
                    _frame.reference_frame_id, _frame.drone_id, P_w_ki.toStr());
      } else {
        vinsframe.odom.pose() = pnp_init.second;
      }
    }
  }

  auto frame_ret = state.addFrame(_frame, vinsframe);
  spdlog::debug("Add Remote VINSFrame with {}: {} IMU {} iskeyframe {}/{}",
                _frame.drone_id, vinsframe.toStr(), _frame.imu_buf.size(),
                vinsframe.is_keyframe, _frame.is_keyframe);
  return frame_ret;
}

void D2Estimator::addSldWinToFrame(VisualImageDescArray &frame) {
  for (unsigned int i = 0; i < state.size(); i++) {
    frame.sld_win_status.push_back(state.getFrame(i).frame_id);
  }
}

void D2Estimator::inputRemoteImage(VisualImageDescArray &frame) {
  const Guard lock(frame_mutex);
  if (!isInitialized() &&
      params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS) {
    // In consenus mode, we require first to be local initialized before
    // deal with remote
    return;
  }
  if (frame.sld_win_status.size() > 0) {
    // We need to update the sliding window.
    updateSldwin(frame.drone_id, frame.sld_win_status);
  }
  auto frame_ptr = addFrameRemote(frame);
  if (params->estimation_mode == D2Common::SERVER_MODE &&
      state.size(frame.drone_id) >= params->min_solve_frames) {
    state.clearUselessFrames();
    solveNonDistrib();
  }
  visual.pubFrame(frame_ptr);
}

bool D2Estimator::inputImage(VisualImageDescArray &_frame) {
  // Guard
  const Guard lock(frame_mutex);
  if (!initFirstPoseFlag) {
    SPDLOG_INFO("tryinitFirstPose imu buf: {}", imu_bufs[self_id].size());
    initFirstPoseFlag = tryinitFirstPose(_frame);
    return initFirstPoseFlag;
  }

  if (!isInitialized() && !_frame.is_keyframe && !_frame.is_stereo &&
      params->enable_sfm_initialization) {
    // Do add when not solved and not keyframe
    return false;
  }

  double t_imu_frame = _frame.stamp + state.getTd();
  while (!imu_bufs[self_id].available(t_imu_frame)) {
    // Wait for IMU
    usleep(2000);
    SPDLOG_WARN("wait for imu...");
  }

  auto frame = addFrame(_frame);
  if (state.size() >= params->min_solve_frames &&
      params->estimation_mode != D2Common::DISTRIBUTED_CAMERA_CONSENUS) {
    solveNonDistrib();
  }
  addSldWinToFrame(_frame);
  frame_count++;
  updated = true;
  if (isInitialized()) {
    visual.pubFrame(frame);
  }
  return true;
}

void D2Estimator::setStateProperties() {
  ceres::Problem &problem = solver->getProblem();
  auto pose_local_param = new PoseLocalParameterization;
  // set LocalParameterization
  for (auto &drone_id : state.availableDrones()) {
    if (state.size(drone_id) > 0) {
      for (size_t i = 0; i < state.size(drone_id); i++) {
        auto frame_a = state.getFrame(drone_id, i);
        auto pointer = state.getPoseState(frame_a.frame_id);
        if (problem.HasParameterBlock(CheckGetPtr(pointer))) {
          problem.SetParameterization(CheckGetPtr(pointer), pose_local_param);
        }
      }
    }
  }

  bool is_first = true;
  for (auto cam_id : state.getAvailableCameraIds()) {
    auto pointer = state.getExtrinsicState(cam_id);
    if (is_first && params->not_estimate_first_extrinsic &&
        state.getCameraBelonging(cam_id) == self_id) {
      if (problem.HasParameterBlock(CheckGetPtr(pointer))) {
        problem.SetParameterBlockConstant(CheckGetPtr(pointer));
        is_first = false;
      } else {
        SPDLOG_CRITICAL(
            "first extrinsic {} not in problem. This should be a bug "
            "or feature tracking is totally wrong. Will print sldwin "
            "for debugging...",
            cam_id);
        state.printSldWin(keyframe_measurements);
      }
    }
    if (!problem.HasParameterBlock(CheckGetPtr(pointer))) {
      continue;
    }
    int drone_id = state.getCameraBelonging(cam_id);
    if (!params->estimate_extrinsic ||
        state.size(drone_id) < params->max_sld_win_size ||
        state.lastFrame().odom.vel().norm() <
            params->estimate_extrinsic_vel_thres) {
      problem.SetParameterBlockConstant(CheckGetPtr(state.getExtrinsicState(cam_id)));
    }
    problem.SetParameterization(CheckGetPtr(state.getExtrinsicState(cam_id)),
                                pose_local_param);
  }

  for (auto lm_id : used_landmarks) {
    auto pointer = state.getLandmarkState(lm_id);
    if (!problem.HasParameterBlock(CheckGetPtr(pointer))) {
      continue;
    }
  }

  if (!params->estimate_td || state.size() < params->max_sld_win_size ||
      state.lastFrame().odom.vel().norm() <
          params->estimate_extrinsic_vel_thres) {
    problem.SetParameterBlockConstant(CheckGetPtr(state.getTdState(self_id)));
  }

  if (!state.getPrior() || params->always_fixed_first_pose) {
    // As we added prior for first pose, we do not need to fix it.
    problem.SetParameterBlockConstant(
        CheckGetPtr(state.getPoseState(state.firstFrame(self_id).frame_id)));
  }
}

bool D2Estimator::isMain() const {
  return self_id == params->main_id;  // Temp code/
}

void D2Estimator::onDistributedVinsData(const DistributedVinsData &dist_data) {
  spdlog::debug("D{} drone {} solver_id {} iteration {} reference_frame_id {}",
                self_id, dist_data.drone_id, dist_data.solver_token,
                dist_data.iteration_count, dist_data.reference_frame_id);
  if (dist_data.reference_frame_id == state.getReferenceFrameId()) {
    sync_data_receiver->add(dist_data);
  }
}

void D2Estimator::onSyncSignal(int drone_id, int signal, int64_t token) {
  if (signal == DSolverReady || signal == DSolverNonDist) {
    ready_drones.insert(drone_id);
  }
  if (signal == DSolverStart || (signal == DSolverNonDist && isMain())) {
    // First drone start or claim non dist
    ready_to_start = true;
    solve_token = token;
  }
  if (isMain() && ready_drones.size() == state.availableDrones().size()) {
    ready_to_start = true;
  }
}

void D2Estimator::sendDistributedVinsData(DistributedVinsData data) {
  data.reference_frame_id = state.getReferenceFrameId();
  if (vinsnet!=nullptr)
  {
    vinsnet->sendDistributedVinsData(data);
  } else {
    SPDLOG_WARN("D{} sendDistributedVinsData but net is nullptr", self_id);
  }
}

void D2Estimator::sendSyncSignal(SyncSignal data, int64_t token) {
  if (vinsnet)
  {
    vinsnet->sendSyncSignal((int)data, token);
  } else {
    SPDLOG_WARN("D{} sendSyncSignal but net is nullptr", self_id);
  }
}

bool D2Estimator::readyForStart() {
  if (state.availableDrones().size() == 1) {
    return true;
  }
  return ready_to_start;
}

void D2Estimator::waitForStart() {
  D2Common::Utility::TicToc timer;
  while (!readyForStart()) {
    sendSyncSignal(SyncSignal::DSolverReady, -1);
    usleep(100);
    if (timer.toc() > params->wait_for_start_timout) {
      break;
    }
  }
  double time = timer.toc();
  spdlog::debug("D{} Wait for start time {:.f}", self_id, timer.toc());
  if (time > 100) {
    SPDLOG_WARN("D{} Wait for start time long: {:.1f}", self_id, timer.toc());
  }
}

void D2Estimator::resetMarginalizer() {
  if (marginalizer != nullptr) {
    delete marginalizer;
  }
  marginalizer = new Marginalizer(&state, state.getPrior());
  state.setMarginalizer(marginalizer);
}

void D2Estimator::solveinDistributedMode() {
  if (!updated) {
    return;
  }
  updated = false;
  D2Common::Utility::TicToc tic;
  if (params->consensus_sync_to_start) {
    if (true) {
      ready_drones = std::set<int>{self_id};
      spdlog::debug("D{} ready, wait for start signal...", self_id);
      waitForStart();
      if (isMain()) {
        solve_token += 1;
        sendSyncSignal(SyncSignal::DSolverStart, solve_token);
      }
      static_cast<ConsensusSolver *>(solver)->setToken(solve_token);
      spdlog::debug("D{} All drones read start solving token {}...", self_id,
                    solve_token);
      ready_to_start = false;
    } else {
      // Claim not use a distribured solver.
      sendSyncSignal(SyncSignal::DSolverNonDist, solve_token);
    }
  } else {
    spdlog::debug("D{} async solve...", self_id);
  }

  const Guard lock(frame_mutex);
  // We do not check update, just do optimization everytime go here
  if (state.size() < params->min_solve_frames) {
    // We do not have enough frames to solve.
    return;
  }

  margined_landmarks = state.clearUselessFrames();  // clear in dist mode.
  resetMarginalizer();
  state.preSolve(imu_bufs);
  solver->reset();

  setupImuFactors();
  setupLandmarkFactors();
  setupPriorFactor();
  if (params->enable_perf_output) {
    SPDLOG_INFO("beforeSolve time cost {:.1f} ms", tic.toc());
  }
  auto report = solver->solve();
  state.syncFromState(used_landmarks);

  // Now do some statistics
  static double sum_time = 0;
  static double sum_iteration = 0;
  static double sum_cost = 0;
  sum_time += report.total_time;
  sum_iteration += report.total_iterations;
  sum_cost += report.final_cost;

  if (params->enable_perf_output) {
    SPDLOG_INFO(
        "D{} average time {:.1f}ms, average time of iter: "
        "{:.1f}ms, average iteration {:.3f}, average cost {:.3f}",
        self_id, sum_time * 1000 / solve_count, sum_time * 1000 / sum_iteration,
        sum_iteration / solve_count, sum_cost / solve_count);
  }

  auto last_odom = state.lastFrame().odom;
  SPDLOG_INFO(
      "D{}({}) {}@ref{} landmarks {}/{} v_mea {}/{} drone_num {} "
      "opti_time {:.1f}ms steps {} td {:.1f}ms",
      self_id, solve_count, last_odom.toStr(), state.getReferenceFrameId(),
      used_landmarks.size(), current_landmark_num, current_measurement_num,
      params->max_solve_measurements, state.availableDrones().size(),
      report.total_time * 1000, report.total_iterations, state.getTd() * 1000);

  // Reprogation
  for (auto drone_id : state.availableDrones()) {
    if (drone_id != self_id &&
        params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS) {
      continue;
    }
    auto _imu =
        imu_bufs[self_id].tail(state.lastFrame(drone_id).stamp + state.getTd());
    std::lock_guard<std::recursive_mutex> lock(imu_prop_lock);
    last_prop_odom[drone_id] = _imu.propagation(state.lastFrame(drone_id));
  }

  visual.postSolve();

  if (params->debug_print_states || params->debug_print_sldwin) {
    state.printSldWin(keyframe_measurements);
  }

  if (!report.succ) {
    std::cout << report.message << std::endl;
    exit(1);
  }
  // exit(0);
  solve_count++;
  if (params->enable_perf_output) {
    SPDLOG_INFO("[D2VINS::solveDist: total time cost {:.1f}ms", tic.toc());
  }
}

void D2Estimator::solveNonDistrib() {
  if (params->enable_sfm_initialization) {
    if (state.numKeyframes() < params->min_solve_frames) {
      SPDLOG_WARN("numKeyframes too less, skip optimization");
      return;
    } else {
      if (!isInitialized()) {
        SPDLOG_INFO("Initialization with {} keyframes", state.numKeyframes());
        if (state.monoInitialization()) {
          SPDLOG_INFO("Mono initialization is success, turn to solve");
        } else {
          SPDLOG_ERROR("Initialization failed, will try later\n");
          return;
        }
      }
    }
  }
  resetMarginalizer();
  state.preSolve(imu_bufs);
  solver->reset();
  setupImuFactors();
  setupLandmarkFactors();
  if (current_measurement_num < params->min_solve_cnt) {
    SPDLOG_WARN("Landmark too less: {}, skip optimization", current_measurement_num);
    return;
  }
  setupPriorFactor();
  setStateProperties();
  SolverReport report = solver->solve();
  state.syncFromState(used_landmarks);

  // Now do some statistics
  static double sum_time = 0;
  static double sum_iteration = 0;
  static double sum_cost = 0;
  sum_time += report.total_time;
  sum_iteration += report.total_iterations;
  sum_cost += report.final_cost;

  if (params->enable_perf_output) {
    SPDLOG_INFO(
        "average time {:.1f}ms, average time of iter: {:.1f}ms, "
        "average iteration {:.3f}, average cost {:.3f}\n",
        sum_time * 1000 / solve_count, sum_time * 1000 / sum_iteration,
        sum_iteration / solve_count, sum_cost / solve_count);
  }

  auto last_odom = state.lastFrame().odom;
  auto Ba = state.lastFrame().Ba;
  auto Bg = state.lastFrame().Bg;
  spdlog::info("C{} landmarks {} {} Ba ({:.2f}, {:.2f}, {:.2f}) Bg ({:.2f}, {:.2f}, {:.2f}) td {:.1f}ms opti_time {:.1f}ms",
              solve_count, current_landmark_num, last_odom.toStr(),
              Ba.x(), Ba.y(), Ba.z(), Bg.x(), Bg.y(), Bg.z(),
              state.getTd() * 1000, report.total_time * 1000);

  // Reprogation
  for (auto drone_id : state.availableDrones()) {
    auto _imu =
        imu_bufs[self_id].tail(state.lastFrame(drone_id).stamp + state.getTd());
    std::lock_guard<std::recursive_mutex> lock(imu_prop_lock);
    last_prop_odom[drone_id] = _imu.propagation(state.lastFrame(drone_id));
  }

  visual.postSolve();

  if (params->debug_print_states || params->debug_print_sldwin) {
    state.printSldWin(keyframe_measurements);
  }

  if (!report.succ) {
    std::cout << report.message << std::endl;
    assert(false && "Optimization failed");
  }
  if (solve_count == 0) {
    // Publish the initialized frames uisng visual.pubFrame
    for (auto frame : state.getSldWin(self_id)) {
      visual.pubFrame(frame);
    }
  }
  solve_count++;
}

void D2Estimator::addIMUFactor(FrameIdType frame_ida, FrameIdType frame_idb,
                               const IntegrationBasePtr& pre_integrations) {
  auto imu_factor = std::make_shared<IMUFactor>(pre_integrations);
  auto info = ImuResInfo::create(imu_factor, frame_ida, frame_idb);
  solver->addResidual(info);
  if (params->always_fixed_first_pose) {
    // At this time we fix the first pose and ignore the margin of this imu
    // factor to achieve better numerical stability
    return;
  }
  marginalizer->addResidualInfo(info);
}

void D2Estimator::setupImuFactors() {
  if (state.size() > 1) {
    for (size_t i = 0; i < state.size() - 1; i++) {
      auto &frame_a = state.getFrame(i);
      auto &frame_b = state.getFrame(i + 1);
      auto pre_integrations = frame_b.pre_integrations;  // Prev to current
      // printf("IMU Factor %d<->%d prev %d\n", frame_a.frame_id,
      // frame_b.frame_id, frame_b.prev_frame_id);
      assert(frame_b.prev_frame_id == frame_a.frame_id &&
             "Wrong prev frame id");
      addIMUFactor(frame_a.frame_id, frame_b.frame_id, pre_integrations);
    }
  }

  // In non-distributed mode, we add IMU factor for each drone
  if (params->estimation_mode == D2Common::SOLVE_ALL_MODE ||
      params->estimation_mode == D2Common::SERVER_MODE) {
    for (auto drone_id : state.availableDrones()) {
      if (drone_id == self_id) {
        continue;
      }
      if (state.size(drone_id) > 1) {
        for (size_t i = 0; i < state.size(drone_id) - 1; i++) {
          auto &frame_a = state.getFrame(drone_id, i);
          auto &frame_b = state.getFrame(drone_id, i + 1);
          auto pre_integrations = frame_b.pre_integrations;  // Prev to current
          if (pre_integrations == nullptr) {
            SPDLOG_WARN("frame {}<->{}@{} pre_integrations is nullptr.",
                        frame_a.frame_id, frame_b.frame_id, drone_id);
            continue;
          }
          assert(frame_b.prev_frame_id == frame_a.frame_id &&
                 "Wrong prev frame id on remote");
          addIMUFactor(frame_a.frame_id, frame_b.frame_id, pre_integrations);
        }
      }
    }
  }
}

bool D2Estimator::hasCommonLandmarkMeasurments() {
  auto lms = state.availableLandmarkMeasurements(
      params->max_solve_cnt, params->max_solve_measurements);
  for (auto lm : lms) {
    if (lm.solver_id == -1 && lm.drone_id != self_id) {
      // This is a internal only remote landmark
      continue;
    }
    if (lm.solver_id > 0 && lm.solver_id != self_id) {
      continue;
    }
    for (auto i = 0; i < lm.track.size(); i++) {
      if (state.getFramebyId(lm.track[i].frame_id)->drone_id != self_id) {
        return true;
      }
    }
  }
  return false;
}

void D2Estimator::setupLandmarkFactors() {
  used_landmarks.clear();
  auto lms = state.availableLandmarkMeasurements(
      params->max_solve_cnt, params->max_solve_measurements);
  current_landmark_num = lms.size();
  current_measurement_num = 0;
  auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);
  keyframe_measurements.clear();
  spdlog::debug("{} landmarks", lms.size());
  // We first count keyframe_measurements
  for (auto lm : lms) {
    LandmarkPerFrame firstObs = lm.track[0];
    keyframe_measurements[firstObs.frame_id]++;
    for (auto i = 1; i < lm.track.size(); i++) {
      auto lm_per_frame = lm.track[i];
      keyframe_measurements[lm_per_frame.frame_id]++;
    }
  }
  // Check the measurements number of each keyframe
  std::set<FrameIdType> ignore_frames;
  for (auto it : keyframe_measurements) {
    auto frame_id = it.first;
    if (it.second < params->min_measurements_per_keyframe) {
      auto frame = state.getFramebyId(frame_id);
      if (frame->drone_id != self_id &&
          params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS) {
        ignore_frames.insert(frame_id);
        SPDLOG_WARN("D{} frame_id {} has only {} measurement, will be skip.",
                    self_id, frame_id, it.second);
      } else {
        SPDLOG_WARN("D{} frame_id {} has only {} measurements.", self_id,
                    frame_id, it.second);
      }
      // Print a landmark report for this frame
      //  state.printLandmarkReport(frame_id);
    }
  }

  for (auto lm : lms) {
    auto lm_id = lm.landmark_id;
    LandmarkPerFrame firstObs = lm.track[0];
    if (ignore_frames.find(firstObs.frame_id) != ignore_frames.end()) {
      continue;
    }
    auto base_camera_id = firstObs.camera_id;
    auto mea0 = firstObs.measurement();
    state.getLandmarkbyId(lm_id).solver_flag = LandmarkSolverFlag::SOLVED;
    if (firstObs.depth_mea && params->fuse_dep &&
        firstObs.depth < params->max_depth_to_fuse &&
        firstObs.depth > params->min_depth_to_fuse) {
      auto f_dep = OneFrameDepth::Create(firstObs.depth);
      auto info =
          DepthResInfo::create(f_dep, loss_function, firstObs.frame_id, lm_id);
      marginalizer->addResidualInfo(info);
      solver->addResidual(info);
      used_landmarks.insert(lm_id);
    }
    current_measurement_num++;
    for (auto i = 1; i < lm.track.size(); i++) {
      auto lm_per_frame = lm.track[i];
      if (ignore_frames.find(lm_per_frame.frame_id) != ignore_frames.end()) {
        continue;
      }
      auto mea1 = lm_per_frame.measurement();
      std::shared_ptr<ResidualInfo> info = nullptr;
      if (lm_per_frame.camera_id == base_camera_id) {
        std::shared_ptr<ceres::CostFunction> f_td = nullptr;
        bool enable_depth_mea = false;
        if (lm_per_frame.depth_mea && params->fuse_dep &&
            lm_per_frame.depth < params->max_depth_to_fuse &&
            lm_per_frame.depth > params->min_depth_to_fuse) {
          enable_depth_mea = true;
          f_td = std::make_shared<ProjectionTwoFrameOneCamDepthFactor>(
              mea0, mea1, firstObs.velocity, lm_per_frame.velocity,
              firstObs.cur_td, lm_per_frame.cur_td, lm_per_frame.depth);
        } else {
          f_td = std::make_shared<ProjectionTwoFrameOneCamFactor>(
              mea0, mea1, firstObs.velocity, lm_per_frame.velocity,
              firstObs.cur_td, lm_per_frame.cur_td);
        }
        if (firstObs.frame_id == lm_per_frame.frame_id) {
          SPDLOG_WARN(
              "landmarkid {} frame {}<->{}@{} is the same "
              "camera id {}",
              lm_per_frame.landmark_id, firstObs.frame_id,
              lm_per_frame.frame_id, lm_id, base_camera_id);
          continue;
        }
        info = LandmarkTwoFrameOneCamResInfo::create(
            f_td, loss_function, firstObs.frame_id, lm_per_frame.frame_id,
            lm_id, firstObs.camera_id, enable_depth_mea);
      } else {
        if (lm_per_frame.frame_id == firstObs.frame_id) {
          auto f_td = std::make_shared<ProjectionOneFrameTwoCamFactor>(
              mea0, mea1, firstObs.velocity, lm_per_frame.velocity,
              firstObs.cur_td, lm_per_frame.cur_td);
          info = LandmarkOneFrameTwoCamResInfo::create(
              f_td, loss_function, firstObs.frame_id, lm_id, firstObs.camera_id,
              lm_per_frame.camera_id);
        } else {
          auto f_td = std::make_shared<ProjectionTwoFrameTwoCamFactor>(
              mea0, mea1, firstObs.velocity, lm_per_frame.velocity,
              firstObs.cur_td, lm_per_frame.cur_td);
          info = LandmarkTwoFrameTwoCamResInfo::create(
              f_td, loss_function, firstObs.frame_id, lm_per_frame.frame_id,
              lm_id, firstObs.camera_id, lm_per_frame.camera_id);
        }
      }
      if (info != nullptr) {
        current_measurement_num++;
        solver->addResidual(info);
        marginalizer->addResidualInfo(info);
        used_landmarks.insert(lm_id);
      }
    }
  }
  spdlog::debug("D{} {} landmarks {} measurements {}", self_id, lms.size(),
                current_measurement_num);
}

const std::map<LandmarkIdType, LandmarkPerId> &D2Estimator::getLandmarkDB()
    const {
  return state.getLandmarkDB();
}

const std::vector<VINSFrame *> &D2Estimator::getSelfSldWin() const {
  return state.getSldWin(self_id);
}

void D2Estimator::setupPriorFactor() {
  auto prior_factor = state.getPrior();
  if (prior_factor != nullptr) {
    // Make a copy of the prior factor
    auto pfactor = std::make_shared<PriorFactor>(*prior_factor);
    auto info = std::make_shared<PriorResInfo>(pfactor);
    solver->addResidual(info);
    marginalizer->addResidualInfo(info);
  }
}

std::vector<LandmarkPerId> D2Estimator::getMarginedLandmarks() const {
  return margined_landmarks;
}

Swarm::Odometry D2Estimator::getImuPropagation() {
  std::lock_guard<std::recursive_mutex> lock(imu_prop_lock);
  return last_prop_odom.at(self_id);
}

Swarm::Odometry D2Estimator::getOdometry() const {
  return getOdometry(self_id);
}

Swarm::Odometry D2Estimator::getOdometry(int drone_id) const {
  // Attention! We output IMU stamp!
  auto odom = state.lastFrame(drone_id).odom;
  odom.stamp = odom.stamp + state.getTd();
  return odom;
}

D2EstimatorState &D2Estimator::getState() { return state; }

bool D2Estimator::isLocalFrame(FrameIdType frame_id) const {
  return state.getFramebyId(frame_id)->drone_id == self_id;
}

D2Visualization &D2Estimator::getVisualizer() { return visual; }

void D2Estimator::setPGOPoses(const std::map<int, Swarm::Pose> &poses) {
  last_pgo_poses = poses;
}

std::set<int> D2Estimator::getNearbyDronesbyPGOData(
    const std::map<int, std::pair<int, Swarm::Pose>> &vins_poses) {
  std::set<int> nearby_drones;
  if (last_pgo_poses.find(self_id) == last_pgo_poses.end()) {
    return nearby_drones;
  }
  auto self_pose = last_pgo_poses.at(self_id);
  for (auto p : last_pgo_poses) {
    if (p.first == self_id) {
      continue;
    }
    auto &pose = p.second;
    auto dist = (pose.pos() - self_pose.pos()).norm();
    auto dist_yaw = std::abs(pose.yaw() - self_pose.yaw());
    if (dist < params->nearby_drone_dist &&
        dist_yaw < params->nearby_drone_yaw_dist / 57.3) {
      nearby_drones.insert(p.first);
    }
    // Check using D2VINS pose
    state.lock_state();
    if (state.size(p.first) > 0) {
      auto d2vins_pose = state.lastFrame(p.first).odom.pose();
      dist = (d2vins_pose.pos() - self_pose.pos()).norm();
      dist_yaw = std::abs(d2vins_pose.yaw() - self_pose.yaw());
    }
    state.unlock_state();
    if (dist < params->nearby_drone_dist &&
        dist_yaw < params->nearby_drone_yaw_dist / 57.3) {
      nearby_drones.insert(p.first);
    }
    spdlog::debug("drone {} dist {:.1f} yaw {:.1f}deg", p.first, dist,
                  dist_yaw * 57.3);
  }
  for (auto it : vins_poses) {
    auto &pose = it.second.second;
    auto dist = (pose.pos() - self_pose.pos()).norm();
    auto dist_yaw = std::abs(pose.yaw() - self_pose.yaw());
    if (dist < params->nearby_drone_dist &&
        dist_yaw < params->nearby_drone_yaw_dist / 57.3) {
      nearby_drones.insert(it.first);
    }
    spdlog::debug("VINS Pose drone {} dist {:.1f} yaw {:.1f}deg", it.first,
                  dist, dist_yaw * 57.3);
  }
  return nearby_drones;
}

std::pair<Swarm::Odometry, std::pair<IMUBuffer, int>>
D2Estimator::getMotionPredict(double stamp) const {
  if (!initFirstPoseFlag) {
    return std::make_pair(Swarm::Odometry(), std::make_pair(IMUBuffer(), -1));
  }
  const auto &last_frame = state.lastFrame();
  auto ret = imu_bufs.at(self_id).periodIMU(last_frame.imu_buf_index,
                                            stamp + state.getTd());
  auto _imu = ret.first;
  auto index = ret.second;
  if (fabs(_imu.size() / (stamp - last_frame.stamp) - params->IMU_FREQ) > 15) {
    SPDLOG_WARN(
        "Local IMU error freq: {:.3f} start_t {:.3f}/{:.3f} end_t "
        "{:.3f}/{:.3f}",
        _imu.size() / (stamp - last_frame.stamp), last_frame.stamp + state.getTd(),
        _imu[0].t, stamp + state.getTd(), _imu[_imu.size() - 1].t);
  }
  return std::make_pair(_imu.propagation(last_frame), ret);
}

void D2Estimator::updateSldwin(int drone_id,
                               const std::vector<FrameIdType> &sld_win) {
  state.updateSldwin(drone_id, sld_win);
}

bool D2Estimator::isInitialized() const { return solve_count > 0; }

}  // namespace D2VINS