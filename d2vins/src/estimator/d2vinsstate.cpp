#include "d2vinsstate.hpp"
#include "../d2vins_params.hpp"
#include <d2common/d2vinsframe.h>
#include <d2common/integration_base.h>
#include "marginalization/marginalization.hpp"
#include "../factors/prior_factor.h"

using namespace Eigen;
using D2Common::generateCameraId;

namespace D2VINS {

D2EstimatorState::D2EstimatorState(int _self_id):
    D2State(_self_id)
{
    sld_wins[self_id] = std::vector<VINSFrame*>();
    if (params->estimation_mode != D2VINSConfig::SERVER_MODE) {
        all_drones.insert(self_id);
    }
}

std::vector<LandmarkPerId> D2EstimatorState::popFrame(int index) {
    const Guard lock(state_lock);
    //Remove from sliding window
    auto frame_id = sld_wins[self_id].at(index)->frame_id;
    if (params->verbose) {
        printf("[D2VSIN::D2EstimatorState] remove frame %ld\n", frame_id);
    }
    sld_wins[self_id].erase(sld_wins[self_id].begin() + index);
    return removeFrameById(frame_id);
}

VINSFrame * D2EstimatorState::addVINSFrame(const VINSFrame & _frame) {
    all_drones.insert(_frame.drone_id);
    const Guard lock(state_lock);
    auto * frame = new VINSFrame;
    *frame = _frame;
    frame_db[frame->frame_id] = frame;
    _frame_pose_state[frame->frame_id] = new state_type[POSE_SIZE];
    _frame.odom.pose().to_vector(_frame_pose_state[frame->frame_id]);
    frame->reference_frame_id = reference_frame_id;
    return frame;
}

std::vector<LandmarkPerId> D2EstimatorState::removeFrameById(FrameIdType frame_id, bool remove_base) {
    const Guard lock(state_lock);
    if (params->verbose) {
        printf("[D2VSIN::D2EstimatorState] remove frame %ld remove base %d\n", frame_id, remove_base);
    }
    auto ret = lmanager.popFrame(frame_id, remove_base);
    auto _frame = static_cast<VINSFrame*>(frame_db.at(frame_id));
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
    for (int i = 0; i < _extrinsic.size(); i ++) {
        auto pose = _extrinsic[i];
        auto cam_id = addCamera(pose, i, self_id);
        local_camera_ids.push_back(cam_id);
    }
    td = _td;
}

CamIdType D2EstimatorState::addCamera(const Swarm::Pose & pose, int camera_index, int drone_id, CamIdType camera_id) {
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
    std::vector<Swarm::Pose> ret;
    for (auto & camera_id : local_camera_ids) {
        ret.push_back(extrinsic.at(camera_id));
    }
    return ret;
}

size_t D2EstimatorState::size() const {
    return size(self_id);
}

VINSFrame & D2EstimatorState::getFrame(int index) {
    return getFrame(self_id, index);
}

const VINSFrame & D2EstimatorState::getFrame(int index) const {
    return getFrame(self_id, index);
}



VINSFrame & D2EstimatorState::firstFrame() {
    return firstFrame(self_id);
}

const VINSFrame & D2EstimatorState::lastFrame() const {
    return lastFrame(self_id);
}

VINSFrame & D2EstimatorState::lastFrame() {
    return lastFrame(self_id);
}

VINSFrame & D2EstimatorState::getFrame(int drone_id, int index) {
    const Guard lock(state_lock);
    return *sld_wins.at(drone_id)[index];
}

const VINSFrame & D2EstimatorState::getFrame(int drone_id, int index) const {
    const Guard lock(state_lock);
    return *sld_wins.at(drone_id)[index];
}


Swarm::Pose D2EstimatorState::getEstimatedPose(int drone_id, int index) const {
    return getFrame(drone_id, index).odom.pose();
}

Swarm::Pose D2EstimatorState::getEstimatedPose(FrameIdType frame_id) const {
    auto drone_id = getFrame(frame_id).drone_id;
    return getFramebyId(frame_id)->odom.pose();
}

Swarm::Odometry D2EstimatorState::getEstimatedOdom(FrameIdType frame_id) const {
    auto drone_id = getFrame(frame_id).drone_id;
    return getFramebyId(frame_id)->odom;
}

VINSFrame & D2EstimatorState::firstFrame(int drone_id) {
    const Guard lock(state_lock);
    assert(sld_wins.at(drone_id).size() > 0 && "SLDWIN size must > 1 to call D2EstimatorState::firstFrame()");
    return *sld_wins.at(drone_id)[0];
}

const VINSFrame & D2EstimatorState::lastFrame(int drone_id) const { 
    const Guard lock(state_lock);
    assert(sld_wins.at(drone_id).size() > 0 && "SLDWIN size must > 1 to call D2EstimatorState::lastFrame()");
    return *sld_wins.at(drone_id).back();
}

VINSFrame & D2EstimatorState::lastFrame(int drone_id) { 
    const Guard lock(state_lock);
    assert(sld_wins.at(drone_id).size() > 0 && "SLDWIN size must > 1 to call D2EstimatorState::lastFrame()");
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

double * D2EstimatorState::getTdState(int drone_id) {
    return &td;
}

double D2EstimatorState::getTd(int drone_id) {
    return td;
}


double * D2EstimatorState::getExtrinsicState(int cam_id) const {
    if (_camera_extrinsic_state.find(cam_id) == _camera_extrinsic_state.end()) {
        printf("[D2VINS::D2EstimatorState] Camera %d not found!\n");
        assert(false && "Camera_id not found");
    }
    return _camera_extrinsic_state.at(cam_id);
}

double * D2EstimatorState::getSpdBiasState(FrameIdType frame_id) const {
    return _frame_spd_Bias_state.at(frame_id);
}

double * D2EstimatorState::getLandmarkState(LandmarkIdType landmark_id) const {
    return lmanager.getLandmarkState(landmark_id);
}

FrameIdType D2EstimatorState::getLandmarkBaseFrame(LandmarkIdType landmark_id) const {
    return lmanager.getLandmarkBaseFrame(landmark_id);
}

Swarm::Pose D2EstimatorState::getExtrinsic(CamIdType cam_id) const {
    return extrinsic.at(cam_id);
}

PriorFactor * D2EstimatorState::getPrior() const {
    return prior_factor;
}

std::set<CamIdType> D2EstimatorState::getAvailableCameraIds() const {
    //Return all camera ids
    std::set<CamIdType> ids;
    for (auto &it : _camera_extrinsic_state) {
        ids.insert(it.first);
    }
    return ids;
}

std::vector<LandmarkPerId> D2EstimatorState::availableLandmarkMeasurements() const {
    return lmanager.availableMeasurements();
}

int D2EstimatorState::getCameraBelonging(CamIdType cam_id) const {
    return camera_drone.at(cam_id);
}

void D2EstimatorState::clearLocalLastNonKeyframe() {
    const Guard lock(state_lock);
    auto & self_sld_win = sld_wins[self_id];
    if (self_sld_win.size() >= params->min_solve_frames) {
        if (!self_sld_win[self_sld_win.size() - 1]->is_keyframe) {
            //If last frame is not keyframe then remove it.
            auto frame_id_to_remove = self_sld_win[self_sld_win.size() - 1]->frame_id;
            if (prior_factor != nullptr) {
                prior_factor->removeFrame(frame_id_to_remove);
            }
            auto tmp = removeFrameById(frame_id_to_remove, false);
            self_sld_win.erase(self_sld_win.end() - 1);
            if (params->verbose) {
                printf("[D2VINS::D2EstimatorState] Remove nonkeyframe %d from local window\n", frame_id_to_remove);
            }
        }
    }
}

std::vector<LandmarkPerId> D2EstimatorState::clearFrame(bool distributed_mode) {
    //If keyframe_only is true, then only remove keyframes.
    const Guard lock(state_lock);
    std::vector<LandmarkPerId> ret;
    std::set<FrameIdType> clear_frames; //Frames in this set will be deleted.
    std::set<FrameIdType> clear_key_frames; //Frames in this set will be MARGINALIZED and deleted.

    for (auto it : latest_remote_sld_wins) {
        auto drone_id = it.first;
        auto & latest_sld_win = it.second;
        std::set<FrameIdType> sld_win_set{latest_sld_win.begin(), latest_sld_win.end()};
        auto & _sld_win = sld_wins.at(drone_id);
        for (auto it : _sld_win) {
            if (sld_win_set.find(it->frame_id) == sld_win_set.end()) {
                clear_frames.insert(it->frame_id);
                if (frame_db.at(it->frame_id)->is_keyframe) {
                    clear_key_frames.insert(it->frame_id);
                }
            }
        }
    }

    // auto & self_sld_win = sld_wins[self_id];
    // int remove_local_num = 0;
    // int remove_index_from_tail = self_sld_win.size() - 2;
    // int remove_index_from_head = 0;
    // while (self_sld_win.size() - remove_local_num > params->max_sld_win_size) {
    //     if (remove_index_from_tail > 0 && !self_sld_win[remove_index_from_tail]->is_keyframe) {
    //         //If last frame is not keyframe then remove it.
    //         clear_frames.insert(self_sld_win[remove_index_from_tail]->frame_id);
    //         remove_index_from_tail --;
    //         remove_local_num++;
    //     } else {
    //         clear_key_frames.insert(self_sld_win[remove_index_from_head]->frame_id);
    //         clear_frames.insert(self_sld_win[remove_index_from_head]->frame_id);
    //         remove_index_from_head ++;
    //         remove_local_num++;
    //     }
    // }
    auto & self_sld_win = sld_wins[self_id];
    if (self_sld_win.size() >= params->min_solve_frames) {
        int count_removed = 0;
        int require_sld_win_size = params->max_sld_win_size;
        int sld_win_size = self_sld_win.size();
        if (distributed_mode) {
            //We remove the second last non keyframe
            if (sld_win_size > require_sld_win_size && !self_sld_win[sld_win_size - 2]->is_keyframe) {
                //Note in distributed mode, after remove the size should be max_sld_win_size
                clear_frames.insert(self_sld_win[sld_win_size - 2]->frame_id);
                count_removed = 1;
            }
        } else {
            require_sld_win_size = params->max_sld_win_size - 1;
            if (sld_win_size > params->max_sld_win_size && !self_sld_win[sld_win_size - 1]->is_keyframe) {
                //If last frame is not keyframe then remove it.
                clear_frames.insert(self_sld_win[sld_win_size - 1]->frame_id);
                count_removed = 1;
            } 
        }
        if (sld_win_size - count_removed > require_sld_win_size) {
            if (self_sld_win[0]->drone_id == self_id) {
                marginalized_self_first = true;
            }
            clear_key_frames.insert(self_sld_win[0]->frame_id);
            clear_frames.insert(self_sld_win[0]->frame_id);
        }
    }

    if (params->enable_marginalization && clear_key_frames.size() > 0) {
        //At this time, non-keyframes is also removed, so add them to remove set to avoid pointer issue.
        clear_key_frames.insert(clear_frames.begin(), clear_frames.end());
        if (marginalizer != nullptr) {
            auto prior_return = marginalizer->marginalize(clear_key_frames);
            if (prior_return!=nullptr) {
                if (prior_factor!=nullptr) {
                    delete prior_factor;
                }
                prior_factor = prior_return;
            }
        }
    }
    if (prior_factor != nullptr) {
        std::vector<ParamInfo> keeps = prior_factor->getKeepParams();
        for (auto p : keeps) {
            if (clear_frames.find(p.id)!=clear_frames.end()) {
                if (params->verbose)
                    printf("[D2EstimatorState::clearFrame] Removed Frame %ld in prior is removed from prior\n", p.id);
                prior_factor->removeFrame(p.id);
            }
        }
    }

    if (clear_frames.size() > 0 ) {
        //Remove frames that are not in the new SLDWIN
        for (auto & _it : sld_wins) {
            auto & _sld_win = _it.second;
            for (auto it = _sld_win.begin(); it != _sld_win.end();) {
                if (clear_frames.find((*it)->frame_id) != clear_frames.end()) {
                    if (params->verbose)
                        printf("[D2EstimatorState::clearFrame] Remove Frame %ld is kf %d\n", (*it)->frame_id, (*it)->is_keyframe);
                    bool remove_base = false;
                    if (clear_key_frames.find((*it)->frame_id) != clear_key_frames.end() && 
                        params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                        //If the frame is a keyframe, then remove the base frame of it's related measurements.
                        //This is because the frame's related measurment's inv_dep is marginalized.
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
    outlierRejection();
    return ret;
}

void D2EstimatorState::updateSldwin(int drone_id, const std::vector<FrameIdType> & sld_win) {
    const Guard lock(state_lock);
    if (params->verbose) {
        printf("[D2VINS::D2EstimatorState] Update SLDWIN for drone %d\n", drone_id);
    }
    if (sld_wins.find(drone_id) == sld_wins.end()) {
        return;
    }
    latest_remote_sld_wins[drone_id] = sld_win;
}

void D2EstimatorState::updateSldWinsIMU(const std::map<int, IMUBuffer> & remote_imu_bufs) {
    if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS || 
        params->estimation_mode == D2VINSConfig::SINGLE_DRONE_MODE) {
        auto & _sld_win = sld_wins[self_id];
        for (size_t i = 0; i < _sld_win.size() - 1; i ++) {
            auto frame_a = _sld_win[i];
            auto frame_b = _sld_win[i+1];
            if (frame_b->prev_frame_id != frame_a->frame_id) {
                //Update IMU factor.
                auto td = getTd(frame_a->drone_id);
                auto ret = remote_imu_bufs.at(self_id).periodIMU(frame_a->imu_buf_index, frame_b->stamp + td);
                auto _imu_buf = ret.first;
                if (frame_b->pre_integrations != nullptr) {
                    delete frame_b->pre_integrations;
                }
                frame_b->pre_integrations = new IntegrationBase(_imu_buf, frame_a->Ba, frame_a->Bg);
                frame_b->prev_frame_id = frame_a->frame_id;
                frame_b->imu_buf_index = ret.second;
                if (fabs(_imu_buf.size()/(frame_b->stamp - frame_a->stamp) - params->IMU_FREQ) > 10) {
                    printf("\033[0;31m[D2VINS::D2Estimator] Remote IMU error freq: %.3f in updateRemoteSldIMU \033[0m\n", 
                        _imu_buf.size()/(frame_b->stamp - frame_a->stamp));
                }
            }
        }
        return;
    }
    for (auto & _it : sld_wins) {
        auto drone_id = _it.first;
        auto & _sld_win = _it.second;
        for (size_t i = 0; i < _sld_win.size() - 1; i ++) {
            auto frame_a = _sld_win[i];
            auto frame_b = _sld_win[i+1];
            if (frame_b->prev_frame_id != frame_a->frame_id) {
                //Update IMU factor.
                auto td = getTd(frame_a->drone_id);
                auto ret = remote_imu_bufs.at(drone_id).periodIMU(frame_a->imu_buf_index, frame_b->stamp + td);
                auto _imu_buf = ret.first;
                frame_b->pre_integrations = new IntegrationBase(_imu_buf, frame_a->Ba, frame_a->Bg);
                frame_b->prev_frame_id = frame_a->frame_id;
                frame_b->imu_buf_index = ret.second;
                if (frame_b->pre_integrations != nullptr) {
                    delete frame_b->pre_integrations;
                }
                if (fabs(_imu_buf.size()/(frame_b->stamp - frame_a->stamp) - params->IMU_FREQ) > 10) {
                    printf("\033[0;31m[D2VINS::D2Estimator] Remote IMU error freq: %.3f in updateRemoteSldIMU \033[0m\n", 
                        _imu_buf.size()/(frame_b->stamp - frame_a->stamp));
                }
            }
        }
    }
}


VINSFrame * D2EstimatorState::addFrame(const VisualImageDescArray & images, const VINSFrame & _frame) {
    const Guard lock(state_lock);
    VINSFrame * frame = addVINSFrame(_frame);
    if (_frame.drone_id != self_id) {
        sld_wins[_frame.drone_id].emplace_back(frame);
        for (auto & img : images.images) {
            if (extrinsic.find(img.camera_id) == extrinsic.end()) {
                printf("[D2VINS::D2EstimatorState] Adding extrinsic of camera %d from drone@%d\n", img.camera_id, _frame.drone_id);
                addCamera(img.extrinsic, img.camera_index, images.drone_id, img.camera_id);
            }
        }
    } else {
        sld_wins[self_id].emplace_back(frame);
    }
    if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS && _frame.drone_id != self_id) {
        //In this mode, the estimate state is always ego-motion and the bias is not been estimated on remote
        _frame.odom.pose().to_vector(_frame_pose_state.at(frame->frame_id));
    } else {
        _frame_spd_Bias_state[frame->frame_id] = new state_type[FRAME_SPDBIAS_SIZE];
        frame->toVector(_frame_pose_state.at(frame->frame_id), _frame_spd_Bias_state.at(frame->frame_id));
    }

    lmanager.addKeyframe(images, td);
    if (params->verbose) {
        printf("[D2VINS::D2EstimatorState%d] add frame %ld@%d ref %d iskeyframe %d with %d images, current %ld frame\n", 
                self_id, images.frame_id, _frame.drone_id, frame->reference_frame_id, frame->is_keyframe, 
                images.images.size(), sld_wins[self_id].size());
    }
    //If first frame we need to add a prior here
    if (size(images.drone_id) == 1 && 
                (images.drone_id == self_id|| params->estimation_mode == D2VINSConfig::SOLVE_ALL_MODE || 
                params->estimation_mode == D2VINSConfig::SERVER_MODE)) {
        //Add a prior for first frame here
        createPriorFactor4FirstFrame(frame);
    }
    return frame;
}

void D2EstimatorState::createPriorFactor4FirstFrame(VINSFrame * frame) {
    //Prior is in form of A \delta x = b
    //A is a 6x6 matrix, A = diag([a_p, a_p, a_p, 0, 0, a_yaw])
    //b is zero vector
    printf("\033[0;32m[D2VINS::D2Estimator] Add prior for first frame\033[0m\n");
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(POSE_EFF_SIZE, POSE_EFF_SIZE);
    A.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * params->initial_pos_sqrt_info;
    A(5, 5) = params->initial_yaw_sqrt_info;
    VectorXd b = VectorXd::Zero(POSE_EFF_SIZE);
    auto param_info = createFramePose(this, frame->frame_id);
    param_info.index = 0;
    std::vector<ParamInfo> params{param_info};
    prior_factor = new PriorFactor(params, A, b);
}

void D2EstimatorState::syncFromState() {
    const Guard lock(state_lock);
    //copy state buffer to structs.
    //First sync the poses

    for (auto it : _frame_pose_state) {
        auto frame_id = it.first;
        if (frame_db.find(frame_id) == frame_db.end()) {
            printf("[D2VINS::D2EstimatorState] Cannot find frame %ld\033[0m\n", frame_id);
        }
        auto frame = static_cast<VINSFrame*>(frame_db.at(frame_id));
        if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS && frame->drone_id != self_id) {
            frame->odom.pose() = Swarm::Pose(it.second);
        }else {
            frame->fromVector(it.second, _frame_spd_Bias_state.at(frame_id));
        }
    }
    for (auto it : _camera_extrinsic_state) {
        auto cam_id = it.first;
        extrinsic.at(cam_id).from_vector(_camera_extrinsic_state.at(cam_id));
    }
    lmanager.syncState(this);
    repropagateIMU();
}

void D2EstimatorState::repropagateIMU() {
    if (sld_wins[self_id].size() > 1) {
        for (size_t i = 0; i < sld_wins[self_id].size() - 1; i ++) {
            auto frame_a = sld_wins[self_id][i];
            auto frame_b = sld_wins[self_id][i+1];
            frame_b->pre_integrations->repropagate(frame_a->Ba, frame_a->Bg);
        }
    }
    if (params->estimation_mode == D2VINSConfig::SOLVE_ALL_MODE) {
        for (auto it : sld_wins) {
            if (it.first == self_id) {
                continue;
            }
            for (size_t i = 0; i < it.second.size() - 1; i ++) {
                auto frame_a = it.second[i];
                auto frame_b = it.second[i+1];
                frame_b->pre_integrations->repropagate(frame_a->Ba, frame_a->Bg);
            }
        }
    }
}

void D2EstimatorState::moveAllPoses(int new_ref_frame_id, const Swarm::Pose & delta_pose) {
    const Guard lock(state_lock);
    reference_frame_id = new_ref_frame_id;
    for (auto it: frame_db) {
        auto frame_id = it.first;
        auto frame = static_cast<VINSFrame*>(it.second);
        frame->moveByPose(new_ref_frame_id, delta_pose);
        if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS && frame->drone_id != self_id) {
            frame->odom.pose().to_vector(_frame_pose_state.at(frame_id));
        } else {
            frame->toVector(_frame_pose_state.at(frame_id), _frame_spd_Bias_state.at(frame_id));
        }
    }
    lmanager.moveByPose(delta_pose);
    if (prior_factor != nullptr) {
        prior_factor->moveByPose(delta_pose);
    }
}

void D2EstimatorState::outlierRejection() {
    //Perform outlier rejection of landmarks
    lmanager.outlierRejection(this);
}

void D2EstimatorState::preSolve(const std::map<int, IMUBuffer> & remote_imu_bufs) {
    updateSldWinsIMU(remote_imu_bufs);
    lmanager.initialLandmarks(this);
}

std::vector<LandmarkPerId> D2EstimatorState::getInitializedLandmarks() const {
    return lmanager.getInitializedLandmarks();
}

LandmarkPerId & D2EstimatorState::getLandmarkbyId(LandmarkIdType id) {
    return lmanager.getLandmark(id);
}

std::vector<LandmarkPerId> D2EstimatorState::getRelatedLandmarks(FrameIdType frame_id) const {
    return lmanager.getRelatedLandmarks(frame_id);
}

bool D2EstimatorState::hasLandmark(LandmarkIdType id) const {
    return lmanager.hasLandmark(id);
}

bool D2EstimatorState::hasCamera(CamIdType frame_id) const {
    return extrinsic.find(frame_id) != extrinsic.end();
}

void D2EstimatorState::printSldWin(const std::map<FrameIdType, int> & keyframe_measurments) const {
    const Guard lock(state_lock);
    for (auto it : sld_wins) {
        printf("=========SLDWIN@drone%d=========\n", it.first);
        for (int i = 0; i < it.second.size(); i ++) {
            int num_mea = 0;
            if (keyframe_measurments.find(it.second[i]->frame_id) != keyframe_measurments.end()) {
                num_mea = keyframe_measurments.at(it.second[i]->frame_id);
            }
            printf("index %d frame_id %ld measurements %d frame: %s\n", i, it.second[i]->frame_id, num_mea, it.second[i]->toStr().c_str());
        }
        printf("========================\n");
    }
}

const std::vector<VINSFrame*> & D2EstimatorState::getSldWin(int drone_id) const {
    return sld_wins.at(self_id);
}

void D2EstimatorState::solveGyroscopeBias() {
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    auto & sld_win = sld_wins[self_id];
    for (int i = 0; i < sld_win.size() - 1; i ++ ) {
        auto frame_i = sld_win[i];
        auto frame_j = sld_win[i + 1];
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->R().transpose() * frame_j->R());
        tmp_A = frame_j->pre_integrations->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->pre_integrations->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    delta_bg = A.ldlt().solve(b);
    printf("[D2EstimatorState] gyroscope bias initial calibration: ");
    std::cout << delta_bg.transpose() << std::endl;

    for (int i = 0; i < sld_win.size() - 1; i++) {
        auto frame_i = sld_win[i];
        auto frame_id = frame_i->frame_id;
        frame_i->Bg += delta_bg;
        frame_i->toVector(_frame_pose_state[frame_id], _frame_spd_Bias_state[frame_id]);
    }

    for (int i = 0; i < sld_win.size() - 1; i++) {
        auto frame_i = sld_win[i];
        frame_i->pre_integrations->repropagate(frame_i->Ba, frame_i->Bg);
    }
}

}