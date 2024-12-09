#include "d2vinsstate.hpp"

#include <d2common/d2vinsframe.h>
#include <d2common/integration_base.h>
#include <spdlog/spdlog.h>

#include <d2common/utils.hpp>

#include "../d2vins_params.hpp"
#include "../factors/prior_factor.h"
#include "marginalization/marginalization.hpp"

using namespace Eigen;
using D2Common::generateCameraId;

namespace D2VINS {

D2EstimatorState::D2EstimatorState(int _self_id) : D2State(_self_id) {
  sld_wins[self_id] = std::vector<VINSFrame *>();
  if (params->estimation_mode != D2Common::SERVER_MODE) {
    all_drones.insert(self_id);
  }
}

std::vector<LandmarkPerId> D2EstimatorState::popFrame(int index) {
  const Guard lock(state_lock);
  // Remove from sliding window
  auto frame_id = sld_wins[self_id].at(index)->frame_id;
  if (params->verbose) {
    printf("[D2VSIN::D2EstimatorState] remove frame %ld\n", frame_id);
  }
  sld_wins[self_id].erase(sld_wins[self_id].begin() + index);
  return removeFrameById(frame_id);
}

VINSFrame *D2EstimatorState::addVINSFrame(const VINSFrame &_frame) {
  const Guard lock(state_lock);
  auto *frame = new VINSFrame;
  *frame = _frame;
  frame_db[frame->frame_id] = frame;
  _frame_pose_state[frame->frame_id] = new state_type[POSE_SIZE];
  _frame.odom.pose().to_vector(_frame_pose_state[frame->frame_id]);
  frame->reference_frame_id = reference_frame_id;
  all_drones.insert(_frame.drone_id);
  return frame;
}

std::vector<LandmarkPerId> D2EstimatorState::removeFrameById(
    FrameIdType frame_id, bool remove_base) {
  const Guard lock(state_lock);
  if (params->verbose) {
    printf("[D2VSIN::D2EstimatorState] remove frame %ld remove base %d\n",
           frame_id, remove_base);
  }
  auto ret = lmanager.popFrame(frame_id, remove_base);
  auto _frame = static_cast<VINSFrame *>(frame_db.at(frame_id));
  if (_frame->pre_integrations) {
    delete _frame->pre_integrations;
  }

  delete _frame;
  frame_db.erase(frame_id);
  delete _frame_pose_state.at(frame_id);
  _frame_pose_state.erase(frame_id);
  if (_frame_spd_Bias_state.find(frame_id) != _frame_spd_Bias_state.end()) {
    delete _frame_spd_Bias_state.at(frame_id);
    _frame_spd_Bias_state.erase(frame_id);
  }
  return ret;
}

void D2EstimatorState::init(std::vector<Swarm::Pose> _extrinsic, double _td) {
  const Guard lock(state_lock);
  for (unsigned int i = 0; i < _extrinsic.size(); i++) {
    auto pose = _extrinsic[i];
    auto cam_id = addCamera(pose, i, self_id);
    local_camera_ids.push_back(cam_id);
  }
  td = _td;
}

CamIdType D2EstimatorState::addCamera(const Swarm::Pose &pose, int camera_index,
                                      int drone_id, CamIdType camera_id) {
  const Guard lock(state_lock);
  if (camera_id < 0) {
    camera_id = generateCameraId(self_id, camera_index);
  }
  auto _p = new state_type[POSE_SIZE];
  pose.to_vector(_p);
  _camera_extrinsic_state[camera_id] = _p;
  extrinsic[camera_id] = pose;
  camera_drone[camera_id] = drone_id;
  return camera_id;
}

std::vector<Swarm::Pose> D2EstimatorState::localCameraExtrinsics() const {
  const Guard lock(state_lock);
  std::vector<Swarm::Pose> ret;
  for (auto &camera_id : local_camera_ids) {
    ret.push_back(extrinsic.at(camera_id));
  }
  return ret;
}

size_t D2EstimatorState::size() const { 
  const Guard lock(state_lock);
  return size(self_id); 
}

VINSFrame &D2EstimatorState::getFrame(int index) {
  const Guard lock(state_lock);
  return getFrame(self_id, index);
}

const VINSFrame &D2EstimatorState::getFrame(int index) const {
  const Guard lock(state_lock);
  return getFrame(self_id, index);
}

VINSFrame &D2EstimatorState::firstFrame() { 
  const Guard lock(state_lock);
  return firstFrame(self_id); 
}

const VINSFrame &D2EstimatorState::lastFrame() const {
  const Guard lock(state_lock);
  return lastFrame(self_id);
}

VINSFrame &D2EstimatorState::lastFrame() {
  const Guard lock(state_lock);
  return lastFrame(self_id);
}

VINSFrame &D2EstimatorState::getFrame(int drone_id, int index) {
  const Guard lock(state_lock);
  return *sld_wins.at(drone_id)[index];
}

const VINSFrame &D2EstimatorState::getFrame(int drone_id, int index) const {
  const Guard lock(state_lock);
  return *sld_wins.at(drone_id)[index];
}

Swarm::Pose D2EstimatorState::getEstimatedPose(int drone_id, int index) const {
  const Guard lock(state_lock);
  return getFrame(drone_id, index).odom.pose();
}

Swarm::Pose D2EstimatorState::getEstimatedPose(FrameIdType frame_id) const {
  const Guard lock(state_lock);
  auto drone_id = getFrame(frame_id).drone_id;
  return getFramebyId(frame_id)->odom.pose();
}

Swarm::Odometry D2EstimatorState::getEstimatedOdom(FrameIdType frame_id) const {
  const Guard lock(state_lock);
  auto drone_id = getFrame(frame_id).drone_id;
  return getFramebyId(frame_id)->odom;
}

VINSFrame &D2EstimatorState::firstFrame(int drone_id) {
  const Guard lock(state_lock);
  assert(sld_wins.at(drone_id).size() > 0 &&
         "SLDWIN size must > 0 to call D2EstimatorState::firstFrame()");
  return *sld_wins.at(drone_id)[0];
}

const VINSFrame &D2EstimatorState::lastFrame(int drone_id) const {
  const Guard lock(state_lock);
  if (sld_wins.at(drone_id).size() == 0) {
    printf("[D2VSIN::D2EstimatorState] lastFrame() sld_wins[%d].size() == 0\n",
           drone_id);
  }
  assert(sld_wins.at(drone_id).size() > 0 &&
         "SLDWIN size must > 0 to call D2EstimatorState::lastFrame()");
  return *sld_wins.at(drone_id).back();
}

VINSFrame &D2EstimatorState::lastFrame(int drone_id) {
  const Guard lock(state_lock);
  assert(sld_wins.at(drone_id).size() > 0 &&
         "SLDWIN size must > 0 to call D2EstimatorState::lastFrame()");
  return *sld_wins.at(drone_id).back();
}

size_t D2EstimatorState::size(int drone_id) const {
  const Guard lock(state_lock);
  if (sld_wins.find(drone_id) == sld_wins.end()) {
    return 0;
  }
  return sld_wins.at(drone_id).size();
}

int D2EstimatorState::getPoseIndex(FrameIdType frame_id) const {
  const Guard lock(state_lock);
  return frame_indices.at(frame_id);
}

double *D2EstimatorState::getTdState(int drone_id) { return &td; }

double D2EstimatorState::getTd(int drone_id) { return td; }

double *D2EstimatorState::getExtrinsicState(int cam_id) const {
  const Guard lock(state_lock);
  if (_camera_extrinsic_state.find(cam_id) == _camera_extrinsic_state.end()) {
    printf("[D2VINS::D2EstimatorState] Camera %d not found!\n");
    assert(false && "Camera_id not found");
  }
  return _camera_extrinsic_state.at(cam_id);
}

double *D2EstimatorState::getSpdBiasState(FrameIdType frame_id) const {
  const Guard lock(state_lock);
  return _frame_spd_Bias_state.at(frame_id);
}

double *D2EstimatorState::getLandmarkState(LandmarkIdType landmark_id) const {
  const Guard lock(state_lock);
  return lmanager.getLandmarkState(landmark_id);
}

FrameIdType D2EstimatorState::getLandmarkBaseFrame(
    LandmarkIdType landmark_id) const {
  const Guard lock(state_lock);
  return lmanager.getLandmarkBaseFrame(landmark_id);
}

Swarm::Pose D2EstimatorState::getExtrinsic(CamIdType cam_id) const {
  const Guard lock(state_lock);
  return extrinsic.at(cam_id);
}

PriorFactor *D2EstimatorState::getPrior() const {
  const Guard lock(state_lock);
  return prior_factor;
}

std::set<CamIdType> D2EstimatorState::getAvailableCameraIds() const {
  const Guard lock(state_lock);
  // Return all camera ids
  std::set<CamIdType> ids;
  for (auto &it : _camera_extrinsic_state) {
    ids.insert(it.first);
  }
  return ids;
}

std::vector<LandmarkPerId> D2EstimatorState::availableLandmarkMeasurements(
    int max_pts, int max_measurement) const {
  const Guard lock(state_lock);
  std::set<FrameIdType> current_frames;
  for (auto &it : sld_wins) {
    for (auto &it2 : it.second) {
      current_frames.insert(it2->frame_id);
    }
  }
  return lmanager.availableMeasurements(max_pts, max_measurement,
                                        current_frames);
}

int D2EstimatorState::getCameraBelonging(CamIdType cam_id) const {
  const Guard lock(state_lock);
  return camera_drone.at(cam_id);
}

std::vector<LandmarkPerId> D2EstimatorState::clearUselessFrames(
    bool marginalization) {
  // If keyframe_only is true, then only remove keyframes.
  const Guard lock(state_lock);
  std::vector<LandmarkPerId> ret;
  std::set<FrameIdType> clear_frames;  // Frames in this set will be deleted.
  std::set<FrameIdType>
      clear_key_frames;  // Frames in this set will be MARGINALIZED and deleted.

  for (auto it : latest_remote_sld_wins) {
    auto drone_id = it.first;
    auto &latest_sld_win = it.second;
    std::set<FrameIdType> sld_win_set{latest_sld_win.begin(),
                                      latest_sld_win.end()};
    auto &_sld_win = sld_wins.at(drone_id);
    for (auto it : _sld_win) {
      if (sld_win_set.find(it->frame_id) == sld_win_set.end()) {
        clear_frames.insert(it->frame_id);
        if (frame_db.at(it->frame_id)->is_keyframe) {
          clear_key_frames.insert(it->frame_id);
        }
      }
    }
  }

  auto &self_sld_win = sld_wins[self_id];
  if (self_sld_win.size() >= params->min_solve_frames) {
    int count_removed = 0;
    int require_sld_win_size = params->max_sld_win_size;
    int sld_win_size = self_sld_win.size();
    // We remove the second last non keyframe
    if (sld_win_size > require_sld_win_size &&
        !self_sld_win[sld_win_size - 3]->is_keyframe) {
      clear_frames.insert(self_sld_win[sld_win_size - 3]->frame_id);
      count_removed = 1;
      // Here we attach the intergation base of the remove frame to the last
      // frame
      IntegrationBase *last_pre_int =
          self_sld_win[sld_win_size - 2]->pre_integrations;
      auto remove_pre = self_sld_win[sld_win_size - 3]->pre_integrations;
      remove_pre->push_back(last_pre_int);
      self_sld_win[sld_win_size - 2]->pre_integrations = remove_pre;
      self_sld_win[sld_win_size - 2]->prev_frame_id =
          self_sld_win[sld_win_size - 3]->prev_frame_id;
      self_sld_win[sld_win_size - 3]->pre_integrations =
          nullptr;  // To avoid delete
      // then we delete the useless last_pre_int
      delete last_pre_int;
    }
    if (sld_win_size - count_removed > require_sld_win_size) {
      clear_key_frames.insert(self_sld_win[0]->frame_id);
      clear_frames.insert(self_sld_win[0]->frame_id);
    }
  }

  if (marginalization) {
    if (params->enable_marginalization && clear_key_frames.size() > 0) {
      if (marginalizer != nullptr) {
        auto prior_return = marginalizer->marginalize(clear_key_frames);
        if (prior_return != nullptr) {
          if (prior_factor != nullptr) {
            delete prior_factor;
          }
          prior_factor = prior_return;
        }
      }
    }
    if (prior_factor != nullptr) {
      // At this time, non-keyframes is also removed, so add them to remove set
      // to avoid pointer issue.
      std::vector<ParamInfo> keeps = prior_factor->getKeepParams();
      for (auto p : keeps) {
        if (clear_frames.find(p.id) != clear_frames.end()) {
          if (params->verbose)
            printf(
                "[D2EstimatorState::clearFrame] Removed Frame %ld in prior is "
                "removed from prior\n",
                p.id);
          prior_factor->removeFrame(p.id);
        }
      }
    }
  }

  if (clear_frames.size() > 0) {
    // Remove frames that are not in the new SLDWIN
    for (auto &_it : sld_wins) {
      auto &_sld_win = _it.second;
      for (auto it = _sld_win.begin(); it != _sld_win.end();) {
        if (clear_frames.find((*it)->frame_id) != clear_frames.end()) {
          if (params->verbose)
            printf("[D2EstimatorState::clearFrame] Remove Frame %ld is kf %d\n",
                   (*it)->frame_id, (*it)->is_keyframe);
          bool remove_base = false;
          if (clear_key_frames.find((*it)->frame_id) !=
                  clear_key_frames.end() &&
              params->landmark_param == D2VINSConfig::LM_INV_DEP) {
            // If the frame is a keyframe, then remove the base frame of it's
            // related measurements. This is because the frame's related
            // measurment's inv_dep is marginalized.
            remove_base = params->remove_base_when_margin_remote == 1;
          }
          auto tmp = removeFrameById((*it)->frame_id, remove_base);
          ret.insert(ret.end(), tmp.begin(), tmp.end());
          // delete *it;
          it = _sld_win.erase(it);
        } else {
          ++it;
        }
      }
    }
  }
  return ret;
}

void D2EstimatorState::updateSldwin(int drone_id,
                                    const std::vector<FrameIdType> &sld_win) {
  const Guard lock(state_lock);
  if (params->verbose) {
    printf("[D2VINS::D2EstimatorState] Update sld win for drone %d:", drone_id);
    for (auto id : sld_win) {
      printf("%ld ", id);
    }
    printf("\n");
  }
  if (sld_wins.find(drone_id) == sld_wins.end()) {
    return;
  }
  latest_remote_sld_wins[drone_id] = sld_win;
}

void D2EstimatorState::updateSldWinsIMU(
    const std::map<int, IMUBuffer> &remote_imu_bufs) {
  const Guard lock(state_lock);
  if (params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS ||
      params->estimation_mode == D2Common::SINGLE_DRONE_MODE) {
    auto &_sld_win = sld_wins[self_id];
    for (size_t i = 0; i < _sld_win.size() - 1; i++) {
      auto frame_a = _sld_win[i];
      auto frame_b = _sld_win[i + 1];
      if (frame_b->prev_frame_id != frame_a->frame_id) {
        // Update IMU factor.
        auto td = getTd(frame_a->drone_id);
        auto ret = remote_imu_bufs.at(self_id).periodIMU(frame_a->imu_buf_index,
                                                         frame_b->stamp + td);
        auto _imu_buf = ret.first;
        if (frame_b->pre_integrations != nullptr) {
          delete frame_b->pre_integrations;
        }
        frame_b->pre_integrations =
            new IntegrationBase(_imu_buf, frame_a->Ba, frame_a->Bg);
        frame_b->prev_frame_id = frame_a->frame_id;
        frame_b->imu_buf_index = ret.second;
        if (fabs(_imu_buf.size() / (frame_b->stamp - frame_a->stamp) -
                 params->IMU_FREQ) > 10) {
          printf(
              "\033[0;31m[D2VINS::D2Estimator] Remote IMU error freq: %.3f in "
              "updateRemoteSldIMU \033[0m\n",
              _imu_buf.size() / (frame_b->stamp - frame_a->stamp));
        }
        printf("[D2VINS] Update IMU for %d<->%d\n", frame_a->frame_id,
               frame_b->frame_id);
      }
    }
    return;
  }
  for (auto &_it : sld_wins) {
    auto drone_id = _it.first;
    auto &_sld_win = _it.second;
    for (size_t i = 0; i < _sld_win.size() - 1; i++) {
      auto frame_a = _sld_win[i];
      auto frame_b = _sld_win[i + 1];
      if (frame_b->prev_frame_id != frame_a->frame_id) {
        // Update IMU factor.
        auto td = getTd(frame_a->drone_id);
        auto ret = remote_imu_bufs.at(drone_id).periodIMU(
            frame_a->imu_buf_index, frame_b->stamp + td);
        auto _imu_buf = ret.first;
        frame_b->pre_integrations =
            new IntegrationBase(_imu_buf, frame_a->Ba, frame_a->Bg);
        frame_b->prev_frame_id = frame_a->frame_id;
        frame_b->imu_buf_index = ret.second;
        if (frame_b->pre_integrations != nullptr) {
          delete frame_b->pre_integrations;
        }
        if (fabs(_imu_buf.size() / (frame_b->stamp - frame_a->stamp) -
                 params->IMU_FREQ) > 10) {
          printf(
              "\033[0;31m[D2VINS::D2Estimator] Remote IMU error freq: %.3f in "
              "updateRemoteSldIMU \033[0m\n",
              _imu_buf.size() / (frame_b->stamp - frame_a->stamp));
        }
      }
    }
  }
}

VINSFrame *D2EstimatorState::addFrame(const VisualImageDescArray &images,
                                      const VINSFrame &_frame) {
  const Guard lock(state_lock);
  VINSFrame *frame = addVINSFrame(_frame);
  if (_frame.drone_id != self_id) {
    if (sld_wins.find(_frame.drone_id) == sld_wins.end()) {
      SPDLOG_INFO("[D2VINS::D2EstimatorState] Add sld_win for remote drone {}",
                  _frame.drone_id);
      sld_wins[_frame.drone_id] = std::vector<VINSFrame *>();
    }
    sld_wins[_frame.drone_id].emplace_back(frame);
    for (auto &img : images.images) {
      if (extrinsic.find(img.camera_id) == extrinsic.end()) {
        SPDLOG_INFO(
            "[D2VINS::D2EstimatorState] Adding extrinsic of camera {} from "
            "drone@{}",
            img.camera_id, _frame.drone_id);
        addCamera(img.extrinsic, img.camera_index, images.drone_id,
                  img.camera_id);
      }
    }
  } else {
    sld_wins[self_id].emplace_back(frame);
  }
  if (params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS &&
      _frame.drone_id != self_id) {
    // In this mode, the estimate state is always ego-motion and the bias is not
    // been estimated on remote
    _frame.odom.pose().to_vector(_frame_pose_state.at(frame->frame_id));
  } else {
    _frame_spd_Bias_state[frame->frame_id] = new state_type[FRAME_SPDBIAS_SIZE];
    frame->toVector(_frame_pose_state.at(frame->frame_id),
                    _frame_spd_Bias_state.at(frame->frame_id));
  }

  lmanager.addKeyframe(images, td);
  spdlog::debug(
      "[D2VINS::D2EstimatorState{}] add frame {}@{} ref {} iskeyframe {} with "
      "{} images, current {} frame\n",
      self_id, images.frame_id, _frame.drone_id, frame->reference_frame_id,
      frame->is_keyframe, images.images.size(), sld_wins[self_id].size());
  // If first frame we need to add a prior here
  if (size(images.drone_id) == 1 &&
      (images.drone_id == self_id ||
       params->estimation_mode == D2Common::SOLVE_ALL_MODE ||
       params->estimation_mode == D2Common::SERVER_MODE)) {
    // Add a prior for first frame here
    createPriorFactor4FirstFrame(frame);
  }
  return frame;
}

void D2EstimatorState::createPriorFactor4FirstFrame(VINSFrame *frame) {
  const Guard lock(state_lock);
  // Prior is in form of A \delta x = b
  // A is a 6x6 matrix, A = diag([a_p, a_p, a_p, 0, 0, a_yaw])
  // b is zero vector
  bool add_vel_ba_prior = params->add_vel_ba_prior;
  int local_cam_num = params->camera_num;
  SPDLOG_WARN("Add prior for first frame, extrinsic {}, {} and speed: {}",
              params->estimate_extrinsic, local_cam_num, add_vel_ba_prior);
  int Adim = POSE_EFF_SIZE +
             (params->estimate_extrinsic ? POSE_EFF_SIZE * local_cam_num : 0) +
             (add_vel_ba_prior ? FRAME_SPDBIAS_SIZE : 0);
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Adim, Adim);
  A.block<3, 3>(0, 0) =
      Eigen::Matrix3d::Identity() * params->initial_pos_sqrt_info;
  A(5, 5) = params->initial_yaw_sqrt_info;
  if (add_vel_ba_prior) {
    A.block<3, 3>(POSE_EFF_SIZE, POSE_EFF_SIZE) =
        Eigen::Matrix3d::Identity() * params->initial_vel_sqrt_info;
    A.block<3, 3>(POSE_EFF_SIZE + 3, POSE_EFF_SIZE + 3) =
        Eigen::Matrix3d::Identity() * params->initial_ba_sqrt_info;
    A.block<3, 3>(POSE_EFF_SIZE + 6, POSE_EFF_SIZE + 6) =
        Eigen::Matrix3d::Identity() * params->initial_bg_sqrt_info;
  }
  if (self_id == params->main_id) {
    A = A * 100;
  }
  VectorXd b = VectorXd::Zero(Adim);
  auto param_info = createFramePose(this, frame->frame_id);
  param_info.index = 0;
  int extrinsic_start_idx = POSE_EFF_SIZE;
  std::vector<ParamInfo> need_fix_params{param_info};
  if (add_vel_ba_prior) {
    auto param_info_spd_bias = createSpeedBias(this, frame->frame_id);
    param_info_spd_bias.index = POSE_EFF_SIZE;
    need_fix_params.emplace_back(param_info_spd_bias);
    extrinsic_start_idx += FRAME_SPDBIAS_SIZE;
  }
  if (params->estimate_extrinsic) {
    for (int i = 0; i < local_cam_num; i++) {
      A.block<3, 3>(i * POSE_EFF_SIZE + extrinsic_start_idx,
                    i * POSE_EFF_SIZE + extrinsic_start_idx) =
          Eigen::Matrix3d::Identity() * params->initial_cam_pos_sqrt_info;
      A.block<3, 3>(i * POSE_EFF_SIZE + 3 + extrinsic_start_idx,
                    i * POSE_EFF_SIZE + 3 + extrinsic_start_idx) =
          Eigen::Matrix3d::Identity() * params->initial_cam_ang_sqrt_info;
      auto param_info = createExtrinsic(this, local_camera_ids[i]);
      param_info.index = need_fix_params.back().index + extrinsic_start_idx;
      need_fix_params.emplace_back(param_info);
    }
  }
  prior_factor = new PriorFactor(need_fix_params, A, b);
}

void D2EstimatorState::syncFromState(
    const std::set<LandmarkIdType> &used_landmarks) {
  const Guard lock(state_lock);
  // copy state buffer to structs.
  // First sync the poses

  for (auto it : _frame_pose_state) {
    auto frame_id = it.first;
    if (frame_db.find(frame_id) == frame_db.end()) {
      SPDLOG_ERROR("[D2VINS::D2EstimatorState] Cannot find frame {}", frame_id);
    }
    auto frame = static_cast<VINSFrame *>(frame_db.at(frame_id));
    if (params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS &&
        frame->drone_id != self_id) {
      frame->odom.pose() = Swarm::Pose(it.second);
    } else {
      frame->fromVector(it.second, _frame_spd_Bias_state.at(frame_id));
    }
  }
  for (auto it : _camera_extrinsic_state) {
    auto cam_id = it.first;
    extrinsic.at(cam_id).from_vector(_camera_extrinsic_state.at(cam_id));
  }
  lmanager.syncState(this);
  if (size() < params->max_sld_win_size) {
    // We only repropagte when sld win is smaller than max, means not full
    // initialized.
    SPDLOG_INFO("[D2VINS] not fully initialized, will repropagte IMU");
    repropagateIMU();
  }
  outlierRejection(used_landmarks);
}

void D2EstimatorState::repropagateIMU() {
  const Guard lock(state_lock);
  if (sld_wins[self_id].size() > 1) {
    for (size_t i = 0; i < sld_wins[self_id].size() - 1; i++) {
      auto frame_a = sld_wins[self_id][i];
      auto frame_b = sld_wins[self_id][i + 1];
      frame_b->pre_integrations->repropagate(frame_a->Ba, frame_a->Bg);
    }
  }
  if (params->estimation_mode == D2Common::SOLVE_ALL_MODE) {
    for (auto it : sld_wins) {
      if (it.first == self_id) {
        continue;
      }
      for (size_t i = 0; i < it.second.size() - 1; i++) {
        auto frame_a = it.second[i];
        auto frame_b = it.second[i + 1];
        frame_b->pre_integrations->repropagate(frame_a->Ba, frame_a->Bg);
      }
    }
  }
}

void D2EstimatorState::moveAllPoses(int new_ref_frame_id,
                                    const Swarm::Pose &delta_pose) {
  const Guard lock(state_lock);
  reference_frame_id = new_ref_frame_id;
  for (auto it : frame_db) {
    auto frame_id = it.first;
    auto frame = static_cast<VINSFrame *>(it.second);
    frame->moveByPose(new_ref_frame_id, delta_pose);
    if (params->estimation_mode == D2Common::DISTRIBUTED_CAMERA_CONSENUS &&
        frame->drone_id != self_id) {
      frame->odom.pose().to_vector(_frame_pose_state.at(frame_id));
    } else {
      frame->toVector(_frame_pose_state.at(frame_id),
                      _frame_spd_Bias_state.at(frame_id));
    }
  }
  lmanager.moveByPose(delta_pose);
  if (prior_factor != nullptr) {
    prior_factor->moveByPose(delta_pose);
  }
}

void D2EstimatorState::outlierRejection(
    const std::set<LandmarkIdType> &used_landmarks) {
  const Guard lock(state_lock);
  // Perform outlier rejection of landmarks
  lmanager.outlierRejection(this, used_landmarks);
}

void D2EstimatorState::preSolve(
    const std::map<int, IMUBuffer> &remote_imu_bufs) {
  const Guard lock(state_lock);
  // updateSldWinsIMU(remote_imu_bufs); Useless when IMU bufs are correctly set
  lmanager.initialLandmarks(this);
}

std::vector<LandmarkPerId> D2EstimatorState::getInitializedLandmarks() const {
  const Guard lock(state_lock);
  return lmanager.getInitializedLandmarks(params->landmark_estimate_tracks);
}

LandmarkPerId &D2EstimatorState::getLandmarkbyId(LandmarkIdType id) {
  const Guard lock(state_lock);
  return lmanager.at(id);
}

bool D2EstimatorState::hasLandmark(LandmarkIdType id) const {
  const Guard lock(state_lock);
  return lmanager.hasLandmark(id);
}

bool D2EstimatorState::hasCamera(CamIdType frame_id) const {
  const Guard lock(state_lock);
  return extrinsic.find(frame_id) != extrinsic.end();
}

int D2EstimatorState::numKeyframes() const {
  const Guard lock(state_lock);
  int ret = 0;
  for (auto it : sld_wins) {
    for (auto frame : it.second) {
      if (frame->is_keyframe) {
        ret++;
      }
    }
  }
  return ret;
}

void D2EstimatorState::printSldWin(
    const std::map<FrameIdType, int> &keyframe_measurments) const {
  const Guard lock(state_lock);
  for (auto it : sld_wins) {
    printf("=========SLDWIN@drone%d=========\n", it.first);
    for (unsigned int i = 0; i < it.second.size(); i++) {
      int num_mea = 0;
      if (keyframe_measurments.find(it.second[i]->frame_id) !=
          keyframe_measurments.end()) {
        num_mea = keyframe_measurments.at(it.second[i]->frame_id);
      }
      printf("index %d frame_id %ld is_kf %d measurements %d frame: %s\n", i,
             it.second[i]->frame_id, it.second[i]->is_keyframe, num_mea,
             it.second[i]->toStr().c_str());
    }
    printf("========================\n");
  }
}

const std::vector<VINSFrame *> &D2EstimatorState::getSldWin(
    int drone_id) const {
  const Guard lock(state_lock);
  return sld_wins.at(self_id);
}

void D2EstimatorState::updateEgoMotion() {
  const Guard lock(state_lock);
  auto &sld_win = sld_wins[self_id];
  for (int i = 0; i < static_cast<int>(sld_win.size()) - 1; i++) {
    auto frame_ptr = sld_win[i];
    auto frame_id = sld_win[i]->frame_id;
    if (ego_motions.find(frame_id) == ego_motions.end()) {
      // We need create ego motion for this frame use the data in sld_win
    }
  }
}

void D2EstimatorState::printLandmarkReport(FrameIdType frame_id) const {
  const Guard lock(state_lock);
  auto related_landmarks = lmanager.getRelatedLandmarks(frame_id);
  printf("Related landmarks of frame %ld:\n");
  for (auto lm_id : related_landmarks) {
    auto lm = lmanager.at(lm_id);
    printf("landmark %ld: flag %d tracks %d solve_by_local %d\n",
           lm.landmark_id, lm.flag, lm.track.size(),
           lm.shouldBeSolve(params->self_id));
  }
  printf("===============================\n");
}

void D2EstimatorState::setPose(FrameIdType frame_id, const Swarm::Pose &pose) {
  const Guard lock(state_lock);
  auto frame = static_cast<VINSFrame *>(frame_db.at(frame_id));
  frame->odom.pose() = pose;
  frame->odom.pose().to_vector(_frame_pose_state.at(frame_id));
}

void D2EstimatorState::setVelocity(FrameIdType frame_id,
                                   const Vector3d &velocity) {
  const Guard lock(state_lock);
  auto frame = static_cast<VINSFrame *>(frame_db.at(frame_id));
  frame->odom.vel() = velocity;
  frame->toVector(_frame_pose_state.at(frame_id),
                  _frame_spd_Bias_state.at(frame_id));
}

void D2EstimatorState::setBias(FrameIdType frame_id, const Vector3d &Ba,
                               const Vector3d &Bg) {
  const Guard lock(state_lock);
  auto frame = static_cast<VINSFrame *>(frame_db.at(frame_id));
  frame->Ba = Ba;
  frame->Bg = Bg;
  frame->toVector(_frame_pose_state.at(frame_id),
                  _frame_spd_Bias_state.at(frame_id));
  frame->pre_integrations->repropagate(Ba, Bg);
}

bool D2EstimatorState::monoInitialization() {
  const Guard lock(state_lock);
  // SFM
  std::map<FrameIdType, int> keyframe_measurments;
  printSldWin(keyframe_measurments);

  if (sld_wins.at(self_id).size() < 5) {
    SPDLOG_WARN(
        "monoInitialization: Not enough frames for mono initialization");
    return false;
  }
  auto sld_win = sld_wins.at(self_id);
  const int camera_idx =
      generateCameraId(self_id, 0);  // Default use camera 0 for initialization.
  auto sfm_poses = lmanager.SFMInitialization(sld_win, camera_idx);
  if (sfm_poses.size() == 0) {
    SPDLOG_WARN("monoInitialization: SFM initialization failed");
    return false;
  }

  // Here we rotate the attitude to IMU but not translation
  Swarm::Pose Tbc = extrinsic.at(camera_idx);
  for (auto &[frame_id, pose] : sfm_poses) {
    sfm_poses[frame_id].att() = pose.att() * Tbc.att().inverse();
  }

  // Then use these poses to perform gyro bias calibration
  if (!solveGyroscopeBias(sld_win, sfm_poses, Tbc)) {
    SPDLOG_WARN("monoInitialization: Gyroscope bias calibration failed");
    return false;
  }
  // Then use these poses to perform linear alignment
  if (!LinearAlignment(sld_win, sfm_poses, Tbc)) {
    SPDLOG_WARN("monoInitialization: Linear alignment failed");
    return false;
  }

  SPDLOG_INFO("monoInitialization: Finished mono initialization");
  return true;
}

bool D2EstimatorState::solveGyroscopeBias(
    std::vector<VINSFrame *> sld_win,
    const std::map<FrameIdType, Swarm::Pose> &sfm_poses,
    Swarm::Pose extrinsic) {
  const Guard lock(state_lock);
  // Migrating from VINS-Mono
  Matrix3d A;
  Vector3d b;
  Vector3d delta_bg;
  A.setZero();
  b.setZero();
  for (int i = 0; i < static_cast<int>(sld_win.size()) - 1; i++) {
    auto frame_i = sld_win[i];
    auto frame_j = sld_win[i + 1];
    MatrixXd tmp_A(3, 3);
    tmp_A.setZero();
    VectorXd tmp_b(3);
    tmp_b.setZero();
    Eigen::Quaterniond q0 = sfm_poses.at(frame_i->frame_id).att();
    Eigen::Quaterniond q1 = sfm_poses.at(frame_j->frame_id).att();

    Eigen::Quaterniond q_ij(q0.inverse() * q1);
    tmp_A = frame_j->pre_integrations->jacobian.template block<3, 3>(O_R, O_BG);
    tmp_b = 2 * (frame_j->pre_integrations->delta_q.inverse() * q_ij).vec();
    A += tmp_A.transpose() * tmp_A;
    b += tmp_A.transpose() * tmp_b;
  }
  delta_bg = A.ldlt().solve(b);
  printf("[D2EstimatorState] gyroscope bias initial calibration: ");
  std::cout << delta_bg.transpose() << std::endl;

  for (int i = 1; i < sld_win.size(); i++) {
    auto frame_i = sld_win[i];
    auto frame_id = frame_i->frame_id;
    setBias(frame_id, frame_i->Ba, frame_i->Bg + delta_bg);
  }
  return true;
}

bool D2EstimatorState::LinearAlignment(
    std::vector<VINSFrame *> sld_win,
    const std::map<FrameIdType, Swarm::Pose> &sfm_poses,
    Swarm::Pose extrinsic) {
  const Guard lock(state_lock);
  // Migrating from VINS-Mono
  int all_frame_count = sld_win.size();
  int n_state = all_frame_count * 3 + 3 + 1;
  Eigen::Vector3d g;
  Eigen::VectorXd x;

  Eigen::MatrixXd A{n_state, n_state};
  A.setZero();
  Eigen::VectorXd b{n_state};
  b.setZero();

  int i = 0;
  for (int i = 0; i < static_cast<int>(sld_win.size()) - 1; i++) {
    auto frame_i = sld_win[i];
    auto frame_j = sld_win[i + 1];
    Swarm::Pose pose_i = sfm_poses.at(frame_i->frame_id);
    Swarm::Pose pose_j = sfm_poses.at(frame_j->frame_id);
    Eigen::Matrix3d R_i = pose_i.R();
    Eigen::Matrix3d R_j = pose_j.R();
    Eigen::Vector3d T_i = pose_i.pos();
    Eigen::Vector3d T_j = pose_j.pos();

    MatrixXd tmp_A(6, 10);
    tmp_A.setZero();
    VectorXd tmp_b(6);
    tmp_b.setZero();

    double dt = frame_j->pre_integrations->sum_dt;

    tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) =
        R_i.transpose() * dt * dt / 2 * Matrix3d::Identity();
    tmp_A.block<3, 1>(0, 9) = R_i.transpose() * (T_j - T_i) / 100.0;
    tmp_b.block<3, 1>(0, 0) = frame_j->pre_integrations->delta_p +
                              R_i.transpose() * R_j * extrinsic.pos() -
                              extrinsic.pos();
    tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = R_i.transpose() * R_j;
    tmp_A.block<3, 3>(3, 6) = R_i.transpose() * dt * Matrix3d::Identity();
    tmp_b.block<3, 1>(3, 0) = frame_j->pre_integrations->delta_v;

    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
    cov_inv.setIdentity();

    MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
    VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }
  A = A * 1000.0;
  b = b * 1000.0;
  x = A.ldlt().solve(b);
  double s = x(n_state - 1) / 100.0;
  g = x.segment<3>(n_state - 4);
  spdlog::debug(
      "LinearAlignment: Scale: {:.3f} g_norm: {:.3f} g {:.3f} {:.3f} {:.3f}", s,
      g.norm(), g.x(), g.y(), g.z());
  if (fabs(g.norm() - IMUData::Gravity.norm()) > 1.0 || s < 0) {
    SPDLOG_WARN(
        "LinearAlignment Failed. Scale or gnorm wrong: g {:.3f} {:.3f} {:.3f} "
        "s {:.3f}",
        g.x(), g.y(), g.z(), s);
    return false;
  }
  RefineGravity(sld_win, sfm_poses, extrinsic, g, x);
  s = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;
  if (s < 0.0) {
    SPDLOG_WARN("LinearAlignment Failed in RefineGravity. Scale wrong");
    return false;
  }

  // Recover camera poses and IMU poses using the scale
  Eigen::Vector3d pos0 = Eigen::Vector3d::Zero();
  Quaterniond q0 = Utility::g2R(g);
  double yaw = quat2eulers(q0 * sfm_poses.at(sld_win[0]->frame_id).att()).z();
  q0 = eulers2quat(Eigen::Vector3d{0, 0, -yaw}) * q0;
  g = q0 * g;
  SPDLOG_INFO("G final {:.4f} {:.4f} {:.4f}", g.x(), g.y(), g.z());

  for (unsigned int i = 0; i < sld_win.size(); i++) {
    auto frame = sld_win[i];
    Swarm::Pose imu_pose = sfm_poses.at(frame->frame_id);
    if (i == 0) {
      pos0 = s * imu_pose.pos() - imu_pose.att() * extrinsic.pos();
    }
    imu_pose.pos() =
        q0 * (imu_pose.pos() * s - imu_pose.att() * extrinsic.pos() - pos0);
    imu_pose.att() = q0 * imu_pose.att();
    // Set the pose of the frame
    Vector3d vel = imu_pose.R() * x.segment<3>(i * 3);
    setPose(frame->frame_id, imu_pose);
    setVelocity(frame->frame_id, vel);
    SPDLOG_INFO("LinearAlignment: F{} IMU_pose {} vel {:.4f} {:.4f} {:.4f}",
                sld_win[i]->frame_id, imu_pose.toStr(), vel.x(), vel.y(),
                vel.z());
  }
  return true;
}

MatrixXd TangentBasis(Vector3d &g0) {
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if (a == tmp) tmp << 1, 0, 0;
  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

void D2EstimatorState::RefineGravity(
    std::vector<VINSFrame *> sld_win,
    const std::map<FrameIdType, Swarm::Pose> &sfm_poses, Swarm::Pose extrinsic,
    Vector3d &g, VectorXd &x) {
  const Guard lock(state_lock);
  Vector3d g0 = g.normalized() * IMUData::Gravity.norm();
  Vector3d lx, ly;
  // VectorXd x;
  int all_frame_count = sld_win.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  for (int k = 0; k < 4; k++) {
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (int i = 0; i < static_cast<int>(sld_win.size()) - 1; i++) {
      auto frame_i = sld_win[i];
      auto frame_j = sld_win[i + 1];
      Swarm::Pose pose_i = sfm_poses.at(frame_i->frame_id);
      Swarm::Pose pose_j = sfm_poses.at(frame_j->frame_id);
      Eigen::Matrix3d R_i = pose_i.R();
      Eigen::Matrix3d R_j = pose_j.R();
      Eigen::Vector3d T_i = pose_i.pos();
      Eigen::Vector3d T_j = pose_j.pos();

      MatrixXd tmp_A(6, 9);
      tmp_A.setZero();
      VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = frame_j->pre_integrations->sum_dt;

      tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      tmp_A.block<3, 2>(0, 6) =
          R_i.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
      tmp_A.block<3, 1>(0, 8) = R_i.transpose() * (T_j - T_i) / 100.0;
      tmp_b.block<3, 1>(0, 0) = frame_j->pre_integrations->delta_p +
                                R_i.transpose() * R_j * extrinsic.pos() -
                                extrinsic.pos() -
                                R_i.transpose() * dt * dt / 2 * g0;

      tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) = R_i.transpose() * R_j;
      tmp_A.block<3, 2>(3, 6) =
          R_i.transpose() * dt * Matrix3d::Identity() * lxly;
      tmp_b.block<3, 1>(3, 0) =
          frame_j->pre_integrations->delta_v -
          R_i.transpose() * dt * Matrix3d::Identity() * g0;

      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
      // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
      // MatrixXd cov_inv = cov.inverse();
      cov_inv.setIdentity();

      MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * IMUData::Gravity.norm();
  }
  g = g0;
  SPDLOG_INFO(
      "RefineGravity: scale {:.3f} g_norm: {:.3f} g {:.3f} {:.3f} {:.3f}",
      x(n_state - 1), g.norm(), g.x(), g.y(), g.z());
}
}  // namespace D2VINS
