#include "marginalization.hpp"

#include <d2common/utils.hpp>

#include "../../factors/imu_factor.h"
#include "../../factors/prior_factor.h"
#include "../../factors/projectionTwoFrameOneCamFactor.h"
#include "../d2vinsstate.hpp"

using namespace D2Common;

namespace D2VINS {
void Marginalizer::addResidualInfo(ResidualInfo* info) {
  residual_info_list.push_back(info);
}

VectorXd Marginalizer::evaluate(SparseMat& J, int eff_residual_size,
                                int eff_param_size) {
  // Then evaluate all residuals
  // Setup Jacobian
  // row: sort by residual_info_list
  // col: sort by params_list
  int cul_res_size = 0;
  std::vector<Eigen::Triplet<state_type>> triplet_list;
  VectorXd residual_vec(eff_residual_size);
  for (auto info : residual_info_list) {
    if (params->margin_enable_fej) {
      // In this case, we need to evaluate the residual with the FEJ state
      auto params = info->paramsList(state);
      if (last_prior != nullptr) {
        last_prior->replacetoPrevLinearizedPoints(params);
      }
      info->Evaluate(params, true);
    } else {
      info->Evaluate(state);
    }
    auto params = info->paramsList(state);
    auto residual_size = info->residualSize();
    residual_vec.segment(cul_res_size, residual_size) = info->residuals;
    if (std::isnan(info->residuals.maxCoeff()) ||
        std::isnan(info->residuals.minCoeff())) {
      SPDLOG_ERROR(
          "[Marginalization] Residual type {} residuals is nan",
          info->residual_type);
      std::cout << info->residuals.transpose() << std::endl;
      continue;
    }
    for (auto param_blk_i = 0; param_blk_i < params.size(); param_blk_i++) {
      auto& J_blk = info->jacobians[param_blk_i];
      // Place this J to row: cul_res_size, col: param_indices
      auto i0 = cul_res_size;
      auto j0 = _params.at(params[param_blk_i].pointer).index;
      auto param_size = params[param_blk_i].size;
      auto blk_eff_param_size = params[param_blk_i].eff_size;
      if (std::isnan(J_blk.maxCoeff()) || std::isnan(J_blk.minCoeff())) {
        SPDLOG_ERROR(
            "[Marginalization] Residual type {} param_blk {} "
            "jacobians is nan",
            info->residual_type, param_blk_i);
        std::cout << J_blk << std::endl;
        continue;
      }

      for (auto i = 0; i < residual_size; i++) {
        for (auto j = 0; j < blk_eff_param_size; j++) {
          // We only copy the eff param part, that is: on tangent space.
          triplet_list.push_back(
              Eigen::Triplet<state_type>(i0 + i, j0 + j, J_blk(i, j)));
        }
      }
    }
    cul_res_size += residual_size;
  }
  J.setFromTriplets(triplet_list.begin(), triplet_list.end());
  return residual_vec;
}

int Marginalizer::filterResiduals() {
  int eff_residual_size = 0;
  for (auto it = residual_info_list.begin(); it != residual_info_list.end();) {
    if ((*it)->relavant(remove_frame_ids)) {
      eff_residual_size += (*it)->residualSize();
      auto param_list = (*it)->paramsList(state);
      for (auto param_ : param_list) {
        if (_params.find(param_.pointer) == _params.end()) {
          _params[param_.pointer] = param_;
          // Check if param should be remove
          bool is_remove = false;
          auto& param = _params[param_.pointer];
          if ((param.type == POSE || param.type == SPEED_BIAS) &&
              remove_frame_ids.find(param.id) != remove_frame_ids.end()) {
            param.is_remove = true;
          }
          if (param.type == LANDMARK) {
            FrameIdType base_frame_id = state->getLandmarkBaseFrame(param.id);
            param.is_remove = false;
            if (remove_frame_ids.find(base_frame_id) !=
                remove_frame_ids.end()) {
              param.is_remove = true;
            } else {
              if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                // printf("[D2VINS::Marginalizer::filterResiduals] landmark %d
                // base frame %d not in remove_frame_ids %ld but will be
                // remove\n", param.id, base_frame_id,
                // *remove_frame_ids.begin());
                param.is_remove = params->remove_base_when_margin_remote;
              }
            }
          }
        }
      }
      it++;
    } else {
      it = residual_info_list.erase(it);
    }
  }
  return eff_residual_size;
}

void Marginalizer::covarianceEstimation(const SparseMat& H) {
  // This covariance estimation is for debug only. It does not estimate the real
  // covariance in marginalization module. To make it estimate real cov, we need
  // to use full jacobian instead part of the problem.
  auto cov = Utility::inverse(H);
  printf("covarianceEstimation\n");
  for (auto param : params_list) {
    if (param.type == POSE) {
      // Print pose covariance
      printf(
          "[D2VINS::Marginalizer::covarianceEstimation] pose %d covariance:\n",
          param.id);
      std::cout << cov.block(param.index, param.index, POSE_EFF_SIZE,
                             POSE_EFF_SIZE)
                << std::endl;
    }
    if (param.type == EXTRINSIC) {
      // Print extrinsic covariance
      printf(
          "[D2VINS::Marginalizer::covarianceEstimation] extrinsic %d "
          "covariance:\n",
          param.id);
      std::cout << cov.block(param.index, param.index, POSE_EFF_SIZE,
                             POSE_EFF_SIZE)
                << std::endl;
    }
    if (param.type == SPEED_BIAS) {
      // Print speed bias covariance
      printf(
          "[D2VINS::Marginalizer::covarianceEstimation] speed bias %d "
          "covariance:\n",
          param.id);
      std::cout << cov.block(param.index, param.index, FRAME_SPDBIAS_SIZE,
                             FRAME_SPDBIAS_SIZE)
                << std::endl;
    }
  }
}

void Marginalizer::showDeltaXofschurComplement(
    std::vector<ParamInfo> keep_params_list, const SparseMatrix<double>& A,
    const Matrix<double, Dynamic, 1>& b) {
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(A);
  VectorXd _b = solver.solve(b);
  for (auto& param : keep_params_list) {
    printf("param id %d type %d size %d \n", param.id, param.type, param.size);
    std::cout << "A^-1.b:"
              << _b.segment(param.index, param.eff_size).transpose()
              << std::endl;
  }
}

PriorFactor* Marginalizer::marginalize(
    std::set<FrameIdType> _remove_frame_ids) {
  Utility::TicToc tic;
  remove_frame_ids = _remove_frame_ids;
  // Clear previous states
  params_list.clear();
  _params.clear();
  // We first remove all factors that does not evolved frame
  if (params->verbose) {
    for (auto frame_id : remove_frame_ids) {
      SPDLOG_INFO("[Marginalization] remove frame {}", frame_id);
    }
  }

  auto eff_residual_size = filterResiduals();

  sortParams();  // sort the parameters
  if (keep_block_size == 0 || remove_state_dim == 0) {
    SPDLOG_INFO(
        "[Marginalization] keep_block_size={} "
        "remove_state_dim {}",
        keep_block_size, remove_state_dim);
    return nullptr;
  }
  int keep_state_dim = total_eff_state_dim - remove_state_dim;
  Utility::TicToc tt;
  SparseMat J(eff_residual_size, total_eff_state_dim);
  auto b = evaluate(J, eff_residual_size, total_eff_state_dim);
  double t_eval = tt.toc();
  SparseMat H = SparseMatrix<double>(J.transpose()) * J;
  VectorXd g =
      J.transpose() * b;  // Ignore -b here and also in prior_factor.cpp
                          // toJacRes to reduce compuation
  if (params->enable_perf_output) {
    SPDLOG_INFO("[D2VINS::marginalize] evaluation {:.1f}ms JtJ cost {:.1f}ms\n", t_eval,
           tt.toc() - t_eval);
  }
  std::vector<ParamInfo> keep_params_list(
      params_list.begin(), params_list.begin() + keep_block_size);
  if (params->margin_enable_fej && last_prior != nullptr) {
    last_prior->replacetoPrevLinearizedPoints(keep_params_list);
  }
  // Compute the schur complement, by sparse LLT.
  PriorFactor* prior = nullptr;
  tt.tic();
  double t_schur = 0.0;
  if (params->margin_sparse_solver) {
    auto Ab = Utility::schurComplement(H, g, keep_state_dim);
    t_schur = tt.toc();
    prior = new PriorFactor(keep_params_list, Ab.first, Ab.second);
  } else {
    auto Ab = Utility::schurComplement(H.toDense(), g, keep_state_dim);
    t_schur = tt.toc();
    prior = new PriorFactor(keep_params_list, Ab.first, Ab.second);
  }
  if (params->enable_perf_output) {
    SPDLOG_INFO(
          "[Marginalization] SchurComplement cost {:.1f}ms newPrior {:.1f}ms",
          t_schur, tt.toc() - t_schur);
  }

  if (params->enable_perf_output || params->verbose) {
    SPDLOG_INFO(
        "[Marginalization] time cost {:.1f}ms frame_id {} "
        "total_eff_state_dim: {} keep_size {} remove size {} "
        "eff_residual_size: {} keep_block_size {}",
        tic.toc(), *remove_frame_ids.begin(), total_eff_state_dim,
        keep_state_dim, remove_state_dim, eff_residual_size, keep_block_size);
  }

  if (params->debug_write_margin_matrix) {
    Utility::writeMatrixtoFile(params->output_folder + "/H.txt", MatrixXd(H));
    Utility::writeMatrixtoFile(params->output_folder + "/g.txt", MatrixXd(g));
  }

  if (prior->hasNan()) {
    SPDLOG_ERROR(
        "[D2VINS::Marginalizer::marginalize] prior has nan");
    return nullptr;
  }
  return prior;
}

void Marginalizer::sortParams() {
  params_list.clear();
  for (auto it : _params) {
    params_list.push_back(it.second);
  }

  std::sort(params_list.begin(), params_list.end(),
            [](const ParamInfo& a, const ParamInfo& b) {
              if (a.is_remove != b.is_remove) {
                return a.is_remove < b.is_remove;
              } else {
                return a.type < b.type;
              }
            });

  total_eff_state_dim = 0;  // here on tangent space
  remove_state_dim = 0;
  keep_block_size = 0;
  for (unsigned i = 0; i < params_list.size(); i++) {
    auto& _param = _params.at(params_list[i].pointer);
    _param.index = total_eff_state_dim;
    total_eff_state_dim += _param.eff_size;
    if (_param.is_remove) {
      remove_state_dim += _param.eff_size;
    } else {
      keep_block_size++;
    }
    params_list[i] = _param;
  }
}

}  // namespace D2VINS