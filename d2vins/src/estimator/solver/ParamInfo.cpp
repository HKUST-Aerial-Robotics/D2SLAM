#include "ParamInfo.hpp"
#include "../d2vinsstate.hpp"
#include "../../d2vins_params.hpp"

namespace D2VINS {
ParamInfo ParamInfo::createFramePose(D2EstimatorState * state, FrameIdType id) {
    ParamInfo info;
    info.pointer = state->getPoseState(id);
    info.index = -1;
    info.size = POSE_SIZE;
    info.eff_size = POSE_EFF_SIZE;
    info.type = POSE;
    info.id = id;
    info.data_copied = new state_type[info.size];
    memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
    return info;
}

ParamInfo ParamInfo::createExtrinsic(D2EstimatorState * state, int camera_id) {
    ParamInfo info;
    info.pointer = state->getExtrinsicState(camera_id);
    info.index = -1;
    info.size = POSE_SIZE;
    info.eff_size = POSE_EFF_SIZE;
    info.type = EXTRINSIC;
    info.id = camera_id;
    info.data_copied = new state_type[info.size];
    memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
    return info;
}

ParamInfo ParamInfo::createLandmark(D2EstimatorState * state, int landmark_id) {
    ParamInfo info;
    info.pointer = state->getLandmarkState(landmark_id);
    info.index = -1;
    if (params->landmark_param == D2VINS::D2VINSConfig::LM_INV_DEP) {
        info.size = INV_DEP_SIZE;
        info.eff_size = INV_DEP_SIZE;
    } else {
        info.size = POS_SIZE;
        info.eff_size = POS_SIZE;
    }
    info.type = LANDMARK;
    info.id = landmark_id;
    info.data_copied = new state_type[info.size];
    memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
    return info;
}

ParamInfo ParamInfo::createSpeedBias(D2EstimatorState * state, FrameIdType id) {
    ParamInfo info;
    info.pointer = state->getSpdBiasState(id);
    info.index = -1;
    info.size = FRAME_SPDBIAS_SIZE;
    info.eff_size = FRAME_SPDBIAS_SIZE;
    info.type = SPEED_BIAS;
    info.id = id;
    info.data_copied = new state_type[info.size];
    memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
    return info;
}

ParamInfo ParamInfo::createTd(D2EstimatorState * state, int camera_id) {
    ParamInfo info;
    info.pointer = state->getTdState(camera_id);
    info.index = -1;
    info.size = TD_SIZE;
    info.eff_size = TD_SIZE;
    info.type = TD;
    info.id = camera_id;
    info.data_copied = new state_type[info.size];
    memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
    return info;
}
}