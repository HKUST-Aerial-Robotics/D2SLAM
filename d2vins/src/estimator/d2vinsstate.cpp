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
    self_id(_self_id)
{
    sld_wins[self_id] = std::vector<VINSFrame*>();
    if (params->estimation_mode != D2VINSConfig::SERVER_MODE) {
        all_drones.insert(self_id);
    }
    if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS) {
        P_w_iks[self_id] = Swarm::Pose::Identity();
        P_w_ik_state[self_id] = new state_type[POSE_SIZE];
        P_w_iks[self_id].to_vector(P_w_ik_state[self_id]);
    }
}

std::vector<LandmarkPerId> D2EstimatorState::popFrame(int index) {
    const Guard lock(state_lock);
    //Remove from sliding window
    auto frame_id = sld_wins[self_id].at(index)->frame_id;
    if (params->verbose) {
        printf("[D2VSIN::D2EstimatorState] remove frame %ld\n", frame_id);
    }
    delete sld_wins[self_id][index];
    sld_wins[self_id].erase(sld_wins[self_id].begin() + index);
    return removeFrameById(frame_id);
}

std::vector<LandmarkPerId> D2EstimatorState::removeFrameById(FrameIdType frame_id, bool remove_base) {
    const Guard lock(state_lock);
    if (params->verbose) {
        printf("[D2VSIN::D2EstimatorState] remove frame %ld remove base %d\n", frame_id, remove_base);
    }
    auto ret = lmanager.popFrame(frame_id, remove_base);
    frame_db.erase(frame_id);
    delete _frame_pose_state.at(frame_id);
    delete _frame_spd_Bias_state.at(frame_id);
    _frame_pose_state.erase(frame_id);
    _frame_spd_Bias_state.erase(frame_id);
    return ret;
}

void D2EstimatorState::init(std::vector<Swarm::Pose> _extrinsic, double _td) {
    for (int i = 0; i < _extrinsic.size(); i ++) {
        auto pose = _extrinsic[i];
        addCamera(pose, i);
    }
    td = _td;
}

void D2EstimatorState::addCamera(const Swarm::Pose & pose, int camera_index, int camera_id) {
    if (camera_id < 0) {
        camera_id = generateCameraId(self_id, camera_index);
    }
    auto _p = new state_type[POSE_SIZE];
    pose.to_vector(_p);
    _camera_extrinsic_state[camera_id] = _p;
    extrinsic[camera_id] = pose;
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

const VINSFrame & D2EstimatorState::getFramebyId(int frame_id) const {
    if (frame_db.find(frame_id) == frame_db.end()) {
        printf("\033[0;31m[D2EstimatorState::getFramebyId] Frame %d not found in database\033[0m\n", frame_id);
        assert(true && "Frame not found in database");
    }
    return *frame_db.at(frame_id);
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

std::set<int> D2EstimatorState::availableDrones() const { 
    //Should return only has common landmarks.
    return all_drones;
}

VINSFrame & D2EstimatorState::getFrame(int drone_id, int index) {
    const Guard lock(state_lock);
    return *sld_wins.at(drone_id)[index];
}

const VINSFrame & D2EstimatorState::getFrame(int drone_id, int index) const {
    const Guard lock(state_lock);
    return *sld_wins.at(drone_id)[index];
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

double * D2EstimatorState::getPoseState(FrameIdType frame_id) const {
    const Guard lock(state_lock);
    if (_frame_pose_state.find(frame_id) == _frame_pose_state.end()) {
        printf("\033[0;31m[D2VINS::D2EstimatorState] frame %ld not found\033[0m\n", frame_id);
        assert(false && "Frame not found");
    }
    return _frame_pose_state.at(frame_id);
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

Swarm::Pose D2EstimatorState::getPwik(int drone_id) const {
    return P_w_iks.at(drone_id);
}

double * D2EstimatorState::getPwikState(int drone_id) const {
    return P_w_ik_state.at(drone_id);
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

std::vector<LandmarkPerId> D2EstimatorState::clearFrame() {
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

    auto & self_sld_win = sld_wins[self_id];
    if (self_sld_win.size() >= params->min_solve_frames) {
        if (!self_sld_win[self_sld_win.size() - 1]->is_keyframe) {
            //If last frame is not keyframe then remove it.
            clear_frames.insert(self_sld_win[self_sld_win.size() - 1]->frame_id);
        } else if (self_sld_win.size() >= params->max_sld_win_size) {
            clear_key_frames.insert(self_sld_win[0]->frame_id);
            clear_frames.insert(self_sld_win[0]->frame_id);
        }
    }

    if (params->enable_marginalization && clear_key_frames.size() > 0) {
        //At this time, non-keyframes is also removed, so add them to remove set to avoid pointer issue.
        clear_key_frames.insert(clear_frames.begin(), clear_frames.end());
        if (prior_factor!=nullptr) {
            delete prior_factor;
        }
        prior_factor = marginalizer->marginalize(clear_key_frames);
    }
    if (prior_factor != nullptr) {
        std::vector<ParamInfo> keeps = prior_factor->getKeepParams();
        for (auto p : keeps) {
            if (clear_frames.find(p.id)!=clear_frames.end()) {
                if (params->verbose)
                    printf("[D2EstimatorState::clearFrame] Removed Frame %ld in prior is removed from prior\n", p.id);
                std::cout << std::endl;
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
    if (params->estimation_mode != D2VINSConfig::SOLVE_ALL_MODE && params->estimation_mode != D2VINSConfig::SERVER_MODE) {
        return;
    }
    for (auto & _it : sld_wins) {
        auto drone_id = _it.first;
        auto & _sld_win = _it.second;
        if (drone_id == self_id || _sld_win.size() <=1 )
            continue;
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
                if (fabs(_imu_buf.size()/(frame_b->stamp - frame_a->stamp) - params->IMU_FREQ) > 10) {
                    printf("\033[0;31m[D2VINS::D2Estimator] Remote IMU error freq: %.3f in updateRemoteSldIMU \033[0m\n", 
                        _imu_buf.size()/(frame_b->stamp - frame_a->stamp));
                }
            }
        }
    }
}


void D2EstimatorState::addFrame(const VisualImageDescArray & images, const VINSFrame & _frame) {
    const Guard lock(state_lock);
    auto * frame = new VINSFrame;
    *frame = _frame;
    if (_frame.drone_id != self_id) {
        all_drones.insert(_frame.drone_id);
        sld_wins[_frame.drone_id].emplace_back(frame);
        for (auto & img : images.images) {
            if (extrinsic.find(img.camera_id) == extrinsic.end()) {
                printf("[D2VINS::D2EstimatorState] Adding extrinsic of camera %d from drone@%d\n", img.camera_id, _frame.drone_id);
                addCamera(img.extrinsic, img.camera_index, img.camera_id);
            }
        }
        if (P_w_iks.find(_frame.drone_id) == P_w_iks.end()) {
            auto P_w_ik = _frame.odom.pose() * _frame.initial_ego_pose.inverse();
            P_w_iks[_frame.drone_id] = P_w_ik;
            P_w_ik_state[_frame.drone_id] = new state_type[POSE_SIZE];
            P_w_ik.to_vector(P_w_ik_state[_frame.drone_id]);
        }
    } else {
        sld_wins[self_id].emplace_back(frame);
    }
    frame_db[frame->frame_id] = frame;
    _frame_pose_state[frame->frame_id] = new state_type[POSE_SIZE];
    _frame_spd_Bias_state[frame->frame_id] = new state_type[FRAME_SPDBIAS_SIZE];
    if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS && _frame.drone_id != self_id) {
        //In this mode, the estimate state is always ego-motion, and the bias is not been estimated on remote
        auto ego_i = P_w_iks[_frame.drone_id].inverse()* _frame.odom.pose();
        ego_i.to_vector(_frame_pose_state[frame->frame_id]);
    } else {
        frame->toVector(_frame_pose_state[frame->frame_id], _frame_spd_Bias_state[frame->frame_id]);
    }

    lmanager.addKeyframe(images, td);
    if (params->verbose) {
        printf("[D2VINS::D2EstimatorState] add frame %ld@%d iskeyframe %d with %d images, current %ld frame\n", 
            images.frame_id, _frame.drone_id, frame->is_keyframe, images.images.size(), sld_wins[self_id].size());
    }
}

void D2EstimatorState::syncFromState() {
    const Guard lock(state_lock);
    //copy state buffer to structs.
    //First sync the poses
    if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS) {
        //Sync the transformation of frames.
        for (auto it: P_w_iks) {
            auto drone_id = it.first;
            if (drone_id == self_id) {
                continue;
            }
            auto & P_w_ik = it.second;
            P_w_iks[drone_id].from_vector(P_w_ik_state[drone_id]);
        }
    }

    for (auto it : _frame_pose_state) {
        auto frame_id = it.first;
        if (frame_db.find(frame_id) == frame_db.end()) {
            printf("[D2VINS::D2EstimatorState] Cannot find frame %ld\033[0m\n", frame_id);
        }
        auto frame = frame_db.at(frame_id);
        if (params->estimation_mode == D2VINSConfig::DISTRIBUTED_CAMERA_CONSENUS && frame->drone_id != self_id) {
            Swarm::Pose ego_i(it.second);
            frame->odom.pose() = P_w_iks[frame->drone_id] * ego_i;
        }else {
            frame->fromVector(it.second, _frame_spd_Bias_state.at(frame_id));
        }
    }
    for (auto it : _camera_extrinsic_state) {
        auto cam_id = it.first;
        extrinsic.at(cam_id).from_vector(_camera_extrinsic_state.at(cam_id));
    }
    lmanager.syncState(this);

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

void D2EstimatorState::printSldWin() const {
    const Guard lock(state_lock);
    for (auto it : sld_wins) {
        printf("=========SLDWIN@drone%d=========\n", it.first);
        for (int i = 0; i < it.second.size(); i ++) {
            printf("index %d frame_id %ld frame: %s\n", i, it.second[i]->frame_id, it.second[i]->toStr().c_str());
        }
        printf("========================\n");
    }
}

}