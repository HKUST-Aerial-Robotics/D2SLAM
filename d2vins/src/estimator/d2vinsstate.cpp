#include "d2vinsstate.hpp"
#include <d2vins/d2vins_params.hpp>
#include <d2vins/d2vins_types.hpp>
#include "../factors/integration_base.h"
#include "marginalize.hpp"

using namespace Eigen;
namespace D2VINS {

void D2EstimatorState::popFrame(int index) {
    //Remove from sliding window
    auto frame_id = sld_win[index]->frame_id;
    if (params->verbose) {
        printf("[D2VSIN::D2EstimatorState] remove frame %ld\n", frame_id);
    }
    delete sld_win[index];
    sld_win.erase(sld_win.begin() + index);
    lmanager.popFrame(frame_id);
    frame_db.erase(frame_id);
    delete _frame_pose_state[frame_id];
    delete _frame_spd_Bias_state[frame_id];
    _frame_pose_state.erase(frame_id);
    _frame_spd_Bias_state.erase(frame_id);
}

void D2EstimatorState::init(std::vector<Swarm::Pose> _extrinsic, double _td) {
    extrinsic = _extrinsic;
    for (auto & pose : extrinsic) {
        auto _p = new state_type[POSE_SIZE];
        pose.to_vector(_p);
        _camera_extrinsic_state.push_back(_p);
    }
    td = _td;
}

size_t D2EstimatorState::size() const {
    return sld_win.size();
}

VINSFrame & D2EstimatorState::getFrame(int index) {
    return *sld_win[index];
}

VINSFrame & D2EstimatorState::firstFrame() {
    return *sld_win[0];
}

int D2EstimatorState::getPoseIndex(FrameIdType frame_id) const {
    return frame_indices.at(frame_id);
}

double * D2EstimatorState::getPoseState(FrameIdType frame_id) const {
    if (_frame_pose_state.find(frame_id) == _frame_pose_state.end()) {
        printf("\033[0;31m[D2VINS::D2EstimatorState] frame %ld not found\033[0m\n", frame_id);
        exit(-1);
    }
    return _frame_pose_state.at(frame_id);
}

double * D2EstimatorState::getTdState(int camera_id) {
    return &td;
}

double * D2EstimatorState::getExtrinsicState(int i) const {
    return _camera_extrinsic_state[i];
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

Swarm::Pose D2EstimatorState::getExtrinsic(int i) const {
    return extrinsic[i];
}

PriorFactor * D2EstimatorState::getPrior() const {
    return prior_factor;
}


std::vector<LandmarkPerId> D2EstimatorState::availableLandmarkMeasurements() const {
    return lmanager.availableMeasurements();
}

void D2EstimatorState::clearFrame() {
    if (sld_win.size() >= 2 && !sld_win[sld_win.size() - 1]->is_keyframe) {
        //If last frame is not keyframe then remove it.
        popFrame(sld_win.size() - 1);
    } else if (sld_win.size() >= params->max_sld_win_size) {
        if (sld_win[sld_win.size() - 2]->is_keyframe) {
            std::set<FrameIdType> clear_frames{sld_win[0]->frame_id};
            prior_factor = marginalizer->marginalize(clear_frames);
            popFrame(0);
        }
    }
    updatePoseIndices();
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
    sld_win.push_back(frame);
    frame_db[frame->frame_id] = frame;
    _frame_pose_state[frame->frame_id] = new state_type[POSE_SIZE];
    _frame_spd_Bias_state[frame->frame_id] = new state_type[FRAME_SPDBIAS_SIZE];
    frame->toVector(_frame_pose_state[frame->frame_id], _frame_spd_Bias_state[frame->frame_id]);

    lmanager.addKeyframe(images, td);
    if (params->verbose) {
        printf("[D2VINS::D2EstimatorState] add frame %ld, current %ld frame\n", images.frame_id, sld_win.size());
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
    for (size_t i = 0; i < extrinsic.size(); i ++ ) {
        extrinsic[i].from_vector(_camera_extrinsic_state[i]);
    }
    for (auto frame : sld_win) {
        if (frame->pre_integrations != nullptr) {
            frame->pre_integrations->repropagate(frame->Ba, frame->Bg);
        }
    }
    lmanager.syncState(extrinsic, frame_db);
    // outlierRejection();
}

void D2EstimatorState::outlierRejection() {
    //Perform outlier rejection of landmarks
    lmanager.outlierRejection(frame_db, extrinsic);
}

void D2EstimatorState::preSolve() {
    for (auto frame : sld_win) {
        if (frame->pre_integrations != nullptr) {
            frame->pre_integrations->repropagate(frame->Ba, frame->Bg);
        }
    }
    lmanager.initialLandmarks(frame_db, extrinsic);
}

VINSFrame D2EstimatorState::lastFrame() const {
    assert(sld_win.size() > 0 && "SLDWIN size must > 1 to call D2EstimatorState::lastFrame()");
    return *sld_win.back();
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