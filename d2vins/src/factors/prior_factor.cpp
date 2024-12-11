#include "prior_factor.h"

#include <iostream>

#include "../estimator/marginalization/marginalization.hpp"

namespace D2VINS {
bool PriorFactor::hasNan() const {
  if (std::isnan(linearized_jac.maxCoeff()) ||
      std::isnan(linearized_res.minCoeff())) {
    printf("\033[0;31m [D2VINS::PriorFactor] linearized_jac has NaN\033[0m\n");
    return true;
  }
  if (std::isnan(linearized_res.maxCoeff()) ||
      std::isnan(linearized_res.minCoeff())) {
    printf("\033[0;31m [D2VINS::PriorFactor] linearized_res has NaN\033[0m\n");
    return true;
  }
  return false;
}

void PriorFactor::removeFrame(int frame_id) {
  int move_idx = 0;
  for (auto it = keep_params_list.begin(); it != keep_params_list.end();) {
    auto &param = *it;
    param.index -= move_idx;
    if (param.id == frame_id && (param.type == ParamsType::POSE ||
                                 param.type == ParamsType::SPEED_BIAS)) {
      Utility::removeRows(linearized_jac, param.index, param.eff_size);
      Utility::removeCols(linearized_jac, param.index, param.eff_size);
      Utility::removeRows(linearized_res, param.index, param.eff_size);
      keep_eff_param_dim -= param.eff_size;
      keep_param_blk_num--;
      move_idx += param.eff_size;
      // printf("\033[0;31m [D2VINS::PriorFactor] remove frame %d type %d remain
      // size %d \033[0m\n", frame_id, param.type, keep_eff_param_dim);
      it = keep_params_list.erase(it);
    } else {
      it++;
    }
  }
  initDims(keep_params_list);
}

bool PriorFactor::Evaluate(double const *const *parameters, double *residuals,
                           double **jacobians) const {
  Eigen::VectorXd dx(keep_eff_param_dim);
  for (int i = 0; i < keep_param_blk_num; i++) {
    auto &info = keep_params_list[i];
    int size =
        info.size;  // Use norminal size instead of tangent space size here.
    int idx = info.index;
    Eigen::Map<const Eigen::VectorXd> x(parameters[i], size);
    Eigen::Map<const Eigen::VectorXd> x0(info.data_copied.data(), size);
    // std::cout << "idx" << idx << "type" << info.type <<"size" << size  <<
    // "keep_eff_param_dim" <<keep_eff_param_dim<< std::endl;
    if (info.type != POSE && info.type != EXTRINSIC) {
      dx.segment(idx, size) = x - x0;
    } else {
      dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
      Eigen::Quaterniond qerr =
          Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
          Eigen::Quaterniond(x(6), x(3), x(4), x(5));
      dx.segment<3>(idx + 3) = 2.0 * Utility::positify(qerr).vec();
      if (!(qerr.w() >= 0)) {
        dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(qerr).vec();
      }
    }
  }
  Eigen::Map<Eigen::VectorXd> res(residuals, keep_eff_param_dim);
  res = linearized_res + linearized_jac * dx;

  if (jacobians) {
    for (int i = 0; i < keep_param_blk_num; i++) {
      if (jacobians[i]) {
        auto &info = keep_params_list[i];
        int size =
            info.size;  // Use norminal size instead of tangent space size here.
        int idx = info.index;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            jacobian(jacobians[i], keep_eff_param_dim, size);
        jacobian.setZero();
        jacobian.leftCols(info.eff_size) =
            linearized_jac.middleCols(idx, info.eff_size);
      }
    }
  }
  return true;
}

void PriorFactor::moveByPose(const Swarm::Pose &delta_pose) {
  for (auto &info : keep_params_list) {
    if (info.type == ParamsType::POSE) {
      // Move the poses in x0
      Swarm::Pose pose0(info.data_copied.data());
      pose0 = delta_pose * pose0;
      pose0.to_vector(info.data_copied.data());
    }
    if (info.type == ParamsType::SPEED_BIAS) {
      // Move the velocity
      Eigen::Map<Vector3d> speed(info.data_copied.data());
      speed = delta_pose.att() * speed;
    }
  }
}

std::vector<ParamInfo> PriorFactor::getKeepParams() const {
  return keep_params_list;
}

int PriorFactor::getEffParamsDim() const {
  int size = 0;
  for (auto &info : keep_params_list) {
    size += info.eff_size;
  }
  return size;
}

void PriorFactor::initDims(const std::vector<ParamInfo> &_keep_params_list) {
  keep_params_list = _keep_params_list;
  keep_param_blk_num = keep_params_list.size();
  keep_eff_param_dim = getEffParamsDim();
  mutable_parameter_block_sizes()->clear();
  for (auto it : keep_params_list) {
    mutable_parameter_block_sizes()->push_back(it.size);
    keep_params_map[it.pointer] = it;
  }
  set_num_residuals(keep_eff_param_dim);
}

std::pair<MatrixXd, VectorXd> toJacRes(const MatrixXd &A_, const VectorXd &b) {
  MatrixXd A = (A_ + A_.transpose()) / 2;
  const double eps = 1e-8;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
  Eigen::VectorXd S =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                          .select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv =
      Eigen::VectorXd((saes2.eigenvalues().array() > eps)
                          .select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

  VectorXd e0 = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
  MatrixXd J_ = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  // Use pre-conditioned from OKVINS
  // https://github.com/ethz-asl/okvis/blob/master/okvis_ceres/src/MarginalizationError.cpp
  // VectorXd  p = (A.diagonal().array() >
  // eps).select(A.diagonal().cwiseSqrt(),1.0e-3); VectorXd  p_inv =
  // p.cwiseInverse(); SelfAdjointEigenSolver<Eigen::MatrixXd>
  // saes(p_inv.asDiagonal() * A  * p_inv.asDiagonal() ); VectorXd  S_ =
  // (saes.eigenvalues().array() > eps).select(
  //         saes.eigenvalues().array(), 0);
  // VectorXd  S_pinv_ = (saes.eigenvalues().array() > eps).select(
  //         saes.eigenvalues().array().inverse(), 0);
  // VectorXd S_sqrt_ = S_.cwiseSqrt();
  // VectorXd S_pinv_sqrt_ = S_pinv_.cwiseSqrt();

  // // assign Jacobian
  // MatrixXd J_ = (p.asDiagonal() * saes.eigenvectors() *
  // (S_sqrt_.asDiagonal())).transpose();

  // // constant error (residual) _e0 := (-pinv(J^T) * _b):
  // Eigen::MatrixXd J_pinv_T = (S_pinv_sqrt_.asDiagonal())
  //     * saes.eigenvectors().transpose()  *p_inv.asDiagonal() ;
  // VectorXd e0 = (-J_pinv_T * b);
  if (params->debug_write_margin_matrix) {
    Utility::writeMatrixtoFile("/home/xuhao/output/A.txt", A);
    Utility::writeMatrixtoFile("/home/xuhao/output/b.txt", b);
    Utility::writeMatrixtoFile("/home/xuhao/output/J.txt", J_);
    Utility::writeMatrixtoFile("/home/xuhao/output/e0.txt", e0);
  }

  return std::make_pair(J_, e0);
}

std::pair<MatrixXd, VectorXd> toJacRes(const SparseMat &A, const VectorXd &b) {
  return toJacRes(A.toDense(), b);
}

void PriorFactor::replacetoPrevLinearizedPoints(
    std::vector<ParamInfo> &params) {
  std::vector<ParamInfo> new_params;
  int count = 0;
  for (ParamInfo &info : params) {
    if (keep_params_map.count(info.pointer) > 0) {
      // Copy the linearized point
      info.data_copied = keep_params_map.at(info.pointer).data_copied;
      count += 1;
      // printf("Id %d type %d, cur_state:\n", info.id, info.type);
      // std::cout << Map<VectorXd>(info.pointer, info.size).transpose() <<
      // std::endl; std::cout << "linearized point:\n" <<
      // info.data_copied.transpose() << std::endl;
    }
  }
  // printf("Marginalization FEJ state num %d\n", count);
}

}  // namespace D2VINS