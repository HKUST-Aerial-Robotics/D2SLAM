#include "d2vinsstate.hpp"
#include "../d2vins_params.hpp"
#include <d2common/d2vinsframe.h>
#include <d2common/integration_base.h>
#include "marginalization/marginalization.hpp"

using namespace Eigen;
using D2Common::generateCameraId;

namespace D2VINS {


std::vector<LandmarkPerId> D2EstimatorState::popFrame(int index) {
    //Remove from sliding window
    auto frame_id = sld_win[index]->frame_id;
    if (params->verbose) {
        printf("[D2VSIN::D2EstimatorState] remove frame %ld\n", frame_id);
    }
    delete sld_win[index];
    sld_win.erase(sld_win.begin() + index);
    auto ret = lmanager.popFrame(frame_id);
    frame_db.erase(frame_id);
    delete _frame_pose_state[frame_id];
    delete _frame_spd_Bias_state[frame_id];
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
    return sld_win.size();
}

VINSFrame & D2EstimatorState::getFrame(int index) {
    return *sld_win[index];
}

const VINSFrame & D2EstimatorState::getFramebyId(int frame_id) const {
    if (frame_db.find(frame_id) == frame_db.end()) {
        printf("\033[0;31m[D2EstimatorState::getFramebyId] Frame %d not found in database\033[0m\n", frame_id);
        assert(true && "Frame not found in database");
    }
    return *frame_db.at(frame_id);
}


VINSFrame & D2EstimatorState::firstFrame() {
    return *sld_win[0];
}

VINSFrame D2EstimatorState::lastFrame() const {
    assert(sld_win.size() > 0 && "SLDWIN size must > 1 to call D2EstimatorState::lastFrame()");
    return *sld_win.back();
}

std::set<int> D2EstimatorState::availableDrones() const { 
    return all_drones;
}

VINSFrame & D2EstimatorState::getRemoteFrame(int drone_id, int index) {
    if (drone_id == self_id) {
        return getFrame(index);
    }
    return *remote_sld_wins.at(drone_id)[index];
}

VINSFrame & D2EstimatorState::firstRemoteFrame(int drone_id) {
    assert(remote_sld_wins.at(drone_id).size() > 0 && "SLDWIN size must > 1 to call D2EstimatorState::firstRemoteFrame()");
    return *remote_sld_wins.at(drone_id)[0];
}

VINSFrame D2EstimatorState::lastRemoteFrame(int drone_id) const { 
    assert(remote_sld_wins.at(drone_id).size() > 0 && "SLDWIN size must > 1 to call D2EstimatorState::lastRemoteFrame()");
    return *sld_win.back();
}

size_t D2EstimatorState::sizeRemote(int drone_id) const { 
    if (drone_id == self_id)
        return size();
    if (remote_sld_wins.find(drone_id) == remote_sld_wins.end()) {
        return 0;
    }
    return remote_sld_wins.at(drone_id).size();
}

int D2EstimatorState::getPoseIndex(FrameIdType frame_id) const {
    return frame_indices.at(frame_id);
}

double * D2EstimatorState::getPoseState(FrameIdType frame_id) const {
    if (_frame_pose_state.find(frame_id) == _frame_pose_state.end()) {
        printf("\033[0;31m[D2VINS::D2EstimatorState] frame %ld not found\033[0m\n", frame_id);
        assert(false && "Frame not found");
    }
    return _frame_pose_state.at(frame_id);
}

double * D2EstimatorState::getTdState(int camera_index) {
    return &td;
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

std::vector<LandmarkPerId> D2EstimatorState::clearFrame() {
    std::vector<LandmarkPerId> ret;
    if (sld_win.size() >= params->min_solve_frames) {
        if (!sld_win[sld_win.size() - 1]->is_keyframe) {
            //If last frame is not keyframe then remove it.
            ret = popFrame(sld_win.size() - 1);
        } else if (sld_win.size() >= params->max_sld_win_size) {
            std::set<FrameIdType> clear_frames{sld_win[0]->frame_id};
            if (params->enable_marginalization) {
                prior_factor = marginalizer->marginalize(clear_frames);
            }
            ret = popFrame(0);
        }
    }
    outlierRejection();
    updatePoseIndices();
    return ret;
}

void D2EstimatorState::updatePoseIndices() {
    frame_indices.clear();
    for (int i = 0; i < sld_win.size(); i++) {
        frame_indices[sld_win[i]->frame_id] = i;
    }
}

void D2EstimatorState::addFrame(const VisualImageDescArray & images, const VINSFrame & _frame, bool is_keyframe) {
    auto * frame = new VINSFrame;
    *frame = _frame;
    if (_frame.drone_id != self_id) {
        all_drones.insert(_frame.drone_id);
        remote_sld_wins[_frame.drone_id].emplace_back(frame);
        for (auto & img : images.images) {
            if (extrinsic.find(img.camera_id) == extrinsic.end()) {
                printf("[D2VINS::D2EstimatorState] Adding extrinsic of camera %d from drone@%d\n", img.camera_id, _frame.drone_id);
                addCamera(img.extrinsic, img.camera_index, img.camera_id);
            }
        }
    } else {
        sld_win.emplace_back(frame);
    }
    frame_db[frame->frame_id] = frame;
    _frame_pose_state[frame->frame_id] = new state_type[POSE_SIZE];
    _frame_spd_Bias_state[frame->frame_id] = new state_type[FRAME_SPDBIAS_SIZE];
    frame->toVector(_frame_pose_state[frame->frame_id], _frame_spd_Bias_state[frame->frame_id]);

    lmanager.addKeyframe(images, td);
    if (params->verbose) {
        printf("[D2VINS::D2EstimatorState] add frame %ld@%d, current %ld frame\n", images.frame_id, _frame.drone_id, sld_win.size());
    }
    updatePoseIndices();
}

void D2EstimatorState::syncFromState() {
    //copy state buffer to structs.
    //First sync the poses
    for (auto it : _frame_pose_state) {
        auto frame_id = it.first;
        frame_db.at(frame_id)->fromVector(it.second, _frame_spd_Bias_state.at(frame_id));
    }
    for (auto it : _camera_extrinsic_state) {
        auto cam_id = it.first;
        extrinsic.at(cam_id).from_vector(_camera_extrinsic_state.at(cam_id));
    }
    lmanager.syncState(this);
    for (auto frame : sld_win) {
        if (frame->pre_integrations != nullptr) {
            frame->pre_integrations->repropagate(frame->Ba, frame->Bg);
        }
    }
}

void D2EstimatorState::outlierRejection() {
    //Perform outlier rejection of landmarks
    lmanager.outlierRejection(this);
}

void D2EstimatorState::preSolve() {
    lmanager.initialLandmarks(this);
}

std::vector<LandmarkPerId> D2EstimatorState::getInitializedLandmarks() const {
    return lmanager.getInitializedLandmarks();
}

LandmarkPerId & D2EstimatorState::getLandmarkbyId(LandmarkIdType id) {
    return lmanager.getLandmark(id);
}

bool D2EstimatorState::hasLandmark(LandmarkIdType id) const {
    return lmanager.hasLandmark(id);
}

void D2EstimatorState::printSldWin() const {
    printf("=========SLDWIN=========\n");
    for (int i = 0; i < sld_win.size(); i ++) {
        printf("index %d frame_id %ld frame: %s\n", i, sld_win[i]->frame_id, sld_win[i]->toStr().c_str());
    }
    printf("========================\n");
}

}