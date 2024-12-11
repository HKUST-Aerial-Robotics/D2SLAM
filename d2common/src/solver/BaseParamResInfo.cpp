#include <d2common/solver/BaseParamResInfo.hpp>

namespace D2Common {
ParamInfo createFramePose(D2State *state, FrameIdType id, bool is_perturb) {
  ParamInfo info;
  if (is_perturb) {
    info.pointer = state->getPerturbState(id);
    info.size = POSE_EFF_SIZE;
    info.type = POSE_PERTURB_6D;
  } else {
    info.type = POSE;
    info.pointer = state->getPoseState(id);
    info.size = POSE_SIZE;
  }
  info.index = -1;
  info.eff_size = POSE_EFF_SIZE;
  info.id = id;
  info.data_copied = Map<VectorXd>(info.getPointer(), info.size);
  return info;
}

ParamInfo createFrameRotMat(D2State *state, FrameIdType id) {
  ParamInfo info;
  info.pointer = state->getRotState(id);
  info.index = -1;
  info.size = ROTMAT_SIZE;
  info.eff_size = ROTMAT_SIZE;
  info.type = ROTMAT;
  info.id = id;
  info.data_copied = Map<VectorXd>(info.getPointer(), info.size);
  return info;
}

ParamInfo createFramePose4D(D2State *state, FrameIdType id) {
  ParamInfo info;
  info.pointer = state->getPoseState(id);
  info.index = -1;
  info.size = POSE4D_SIZE;
  info.eff_size = POSE4D_SIZE;
  info.type = POSE_4D;
  info.id = id;
  info.data_copied = Map<VectorXd>(info.getPointer(), info.size);
  return info;
}

void ResidualInfo::Evaluate(const std::vector<ParamInfo> &param_infos,
                            bool use_copied) {
  std::vector<const state_type *> params;
  if (use_copied) {
    for (auto &info : param_infos) {
      params.push_back(info.data_copied.data());
    }
  } else {
    for (auto info : param_infos) {
      params.push_back(CheckGetPtr(info.pointer));
    }
  }

  // This function is from VINS.
  residuals.resize(cost_function->num_residuals());
  std::vector<int> blk_sizes = cost_function->parameter_block_sizes();
  std::vector<double *> raw_jacobians(blk_sizes.size());
  jacobians.resize(blk_sizes.size());
  for (unsigned int i = 0; i < blk_sizes.size(); i++) {
    jacobians[i].resize(cost_function->num_residuals(), blk_sizes[i]);
    jacobians[i].setZero();
    raw_jacobians[i] = jacobians[i].data();
  }
  cost_function->Evaluate(params.data(), residuals.data(),
                          raw_jacobians.data());
  if (loss_function) {
    double residual_scaling_, alpha_sq_norm_;
    double sq_norm, rho[3];
    sq_norm = residuals.squaredNorm();
    loss_function->Evaluate(sq_norm, rho);
    double sqrt_rho1_ = sqrt(rho[1]);
    if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
      residual_scaling_ = sqrt_rho1_;
      alpha_sq_norm_ = 0.0;
    } else {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      residual_scaling_ = sqrt_rho1_ / (1 - alpha);
      alpha_sq_norm_ = alpha / sq_norm;
    }
    for (unsigned int i = 0; i < params.size(); i++) {
      jacobians[i] = sqrt_rho1_ * (jacobians[i] -
                                   alpha_sq_norm_ * residuals *
                                       (residuals.transpose() * jacobians[i]));
    }
    residuals *= residual_scaling_;
  }
}

void ResidualInfo::Evaluate(D2State *state) {
  auto param_infos = paramsList(state);
  Evaluate(param_infos);
}

}  // namespace D2Common