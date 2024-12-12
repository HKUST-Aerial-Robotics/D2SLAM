#include "ParamResidualInfo.hpp"

#include "../factors/prior_factor.h"
#include "d2vinsstate.hpp"

namespace D2VINS {

PriorResInfo::PriorResInfo(const PriorFactorPtr& _factor) : ResidualInfo(PriorResidual) {
  cost_function = _factor;
  factor = _factor;
}

bool PriorResInfo::relavant(const std::set<FrameIdType>& frame_ids) const {
  // Prior relavant to all frames.
  return true;
}

std::vector<ParamInfo> PriorResInfo::paramsList(D2State* state) const {
  return factor->getKeepParams();
}

ParamInfo createExtrinsic(D2EstimatorState* state, int camera_id) {
  ParamInfo info;
  info.pointer = state->getExtrinsicState(camera_id);
  info.index = -1;
  info.size = POSE_SIZE;
  info.eff_size = POSE_EFF_SIZE;
  info.type = EXTRINSIC;
  info.id = camera_id;
  info.data_copied = Map<VectorXd>(CheckGetPtr(info.pointer), info.size);
  return info;
}

ParamInfo createLandmark(D2EstimatorState* state, int landmark_id,
                         bool inv_dep_param) {
  ParamInfo info;
  info.pointer = state->getLandmarkState(landmark_id);
  info.index = -1;
  if (inv_dep_param) {
    info.size = INV_DEP_SIZE;
    info.eff_size = INV_DEP_SIZE;
  } else {
    info.size = POS_SIZE;
    info.eff_size = POS_SIZE;
  }
  info.type = LANDMARK;
  info.id = landmark_id;
  info.data_copied = Map<VectorXd>(CheckGetPtr(info.pointer), info.size);
  return info;
}

ParamInfo createSpeedBias(D2EstimatorState* state, FrameIdType id) {
  ParamInfo info;
  info.pointer = state->getSpdBiasState(id);
  info.index = -1;
  info.size = FRAME_SPDBIAS_SIZE;
  info.eff_size = FRAME_SPDBIAS_SIZE;
  info.type = SPEED_BIAS;
  info.id = id;
  info.data_copied = Map<VectorXd>(CheckGetPtr(info.pointer), info.size);
  return info;
}

ParamInfo createTd(D2EstimatorState* state, int camera_id) {
  ParamInfo info;
  info.pointer = state->getTdState(camera_id);
  info.index = -1;
  info.size = TD_SIZE;
  info.eff_size = TD_SIZE;
  info.type = TD;
  info.id = camera_id;
  info.data_copied = Map<VectorXd>(CheckGetPtr(info.pointer), info.size);
  return info;
}

}  // namespace D2VINS