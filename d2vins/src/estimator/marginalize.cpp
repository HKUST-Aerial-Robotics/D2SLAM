#include "marginalize.hpp"
#include <d2vins/utils.hpp>
#include "../factors/prior_factor.h"

namespace D2VINS {
void Marginalizer::addLandmarkResidual(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
    FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id, bool has_td) {
    if (!has_td) {
        auto * info = new LandmarkTwoFrameOneCamResInfo();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->landmark_id = landmark_id;
        info->camera_id = camera_id;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        residual_info_list.push_back(info);
    } else {
        auto * info = new LandmarkTwoFrameOneCamResInfoTD();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->landmark_id = landmark_id;
        info->camera_id = camera_id;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        residual_info_list.push_back(info);
    }
}

void Marginalizer::addImuResidual(ceres::CostFunction * cost_function,  FrameIdType frame_ida, FrameIdType frame_idb) {
    auto * info = new ImuResInfo();
    info->frame_ida = frame_ida;
    info->frame_idb = frame_idb;
    info->cost_function = cost_function;
    info->loss_function = nullptr;
    residual_info_list.push_back(info);
}

void Marginalizer::addPrior(PriorFactor * cost_function) {
    auto * info = new PriorResInfo(cost_function);
    info->cost_function = cost_function;
    info->loss_function = nullptr;
    residual_info_list.push_back(info);
}

VectorXd Marginalizer::evaluate(SparseMat & J, int eff_residual_size, int eff_param_size) {
    //Then evaluate all residuals
    //Setup Jacobian
    //row: sort by residual_info_list
    //col: sort by params_list
    int cul_res_size = 0;
    std::vector<Eigen::Triplet<state_type>> triplet_list;
    VectorXd residual_vec(eff_residual_size);
    for (auto info : residual_info_list) {
        info->Evaluate(state);
        auto params = info->paramsList(state);
        auto residual_size = info->residualSize();
        residual_vec.segment(cul_res_size, residual_size) = info->residuals;
        for (auto param_blk_i = 0; param_blk_i < params.size(); param_blk_i ++) {
            auto & J_blk = info->jacobians[param_blk_i];
            //Place this J to row: cul_res_size, col: param_indices
            auto i0 = cul_res_size;
            auto j0 = _params.at(params[param_blk_i].pointer).index;
            auto param_size = params[param_blk_i].size;
            auto blk_eff_param_size = params[param_blk_i].eff_size;
            for (auto i = 0; i < residual_size; i ++) {
                for (auto j = 0; j < blk_eff_param_size; j ++) {
                    //We only copy the eff param part, that is: on tangent space.
                    triplet_list.push_back(Eigen::Triplet<state_type>(i0 + i, j0 + j, J_blk(i, j)));
                    // printf("J[%d, %d] = %f\n", i0 + i, j0 + j, J_blk(i, j));
                    if (i0 + i >= eff_residual_size || j0 + j >= eff_param_size) {
                        printf("J0 %d\n", j0);
                        printf("J[%d, %d] = %f size %d, %d\n", i0 + i, j0 + j, J_blk(i, j), eff_residual_size, eff_param_size);
                        fflush(stdout);
                        exit(1);
                    }
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
            for (auto param : param_list) {
                //Check if param should be remove
                bool is_remove = false;
                if ((param.type == POSE || param.type == SPEED_BIAS) && remove_frame_ids.find(param.id) != remove_frame_ids.end()) {
                    is_remove = true;
                }
                if (param.type == LANDMARK) {
                    FrameIdType base_frame_id = state->getLandmarkBaseFrame(param.id);
                    is_remove = false;
                    if (remove_frame_ids.find(base_frame_id) != remove_frame_ids.end()) {
                        is_remove = true;
                    } else {
                        printf("[D2VINS::Marginalizer::filterResiduals] landmark %d base frame %d not in remove_frame_ids %ld\n", param.id, base_frame_id, *remove_frame_ids.begin());
                    }
                }
                param.is_remove = is_remove;
                _params[param.pointer] = param;
            }
            it++;
        } else {
            delete *it;
            it = residual_info_list.erase(it);
        }
    }
    return eff_residual_size;
}

PriorFactor * Marginalizer::marginalize(std::set<FrameIdType> _remove_frame_ids) {
    TicToc tic;
    remove_frame_ids = _remove_frame_ids;
    //Clear previous states
    params_list.clear();
    _params.clear();
    //We first remove all factors that does not evolved frame

    auto eff_residual_size = filterResiduals();
    
    sortParams(); //sort the parameters
    
    int keep_state_dim = total_eff_state_dim - remove_state_dim;

    printf("[D2VINS::Marginalizer::marginalize] frame_id %ld total_eff_state_dim: %d remove param size %d eff_residual_size: %d keep_block_size %d \n", 
         *remove_frame_ids.begin(), total_eff_state_dim, remove_state_dim, eff_residual_size, keep_block_size);
    SparseMat J(eff_residual_size, total_eff_state_dim);
    auto b = evaluate(J, eff_residual_size, total_eff_state_dim);
    SparseMat H = SparseMatrix<double>(J.transpose())*J;
    
    std::vector<ParamInfo> keep_params_list(params_list.begin(), params_list.begin() + keep_block_size);
    //Compute the schur complement, by sparse LLT.
    PriorFactor * prior = nullptr;
    if (params->margin_sparse_solver) {
        printf("[D2VINS::Marginalizer::marginalize] use sparse LLT solver\n");
        auto A = Utility::schurComplement(H, b, keep_state_dim);
        prior = new PriorFactor(keep_params_list, A, b);

    } else {
        printf("[D2VINS::Marginalizer::marginalize] use dense solver\n");
        auto A = Utility::schurComplement(H.toDense(), b, keep_state_dim);
        prior = new PriorFactor(keep_params_list, A, b);
    }
    
    printf("[D2VINS::Marginalizer] time cost %.1fms\n", tic.toc());
    return prior;
}

void Marginalizer::sortParams() {
    params_list.clear();
    for (auto it: _params) {
        params_list.push_back(it.second);
    }

    std::sort(params_list.begin(), params_list.end(), [](const ParamInfo & a, const ParamInfo & b) {
        return a.is_remove < b.is_remove;
    });

    total_eff_state_dim = 0; //here on tangent space
    remove_state_dim = 0;
    keep_block_size = 0;
    for (unsigned i = 0; i < params_list.size(); i++) {
        auto & _param = _params.at(params_list[i].pointer);
        _param.index = total_eff_state_dim;
        total_eff_state_dim += _param.eff_size;
        // printf("Param %p type %d size %d index %d cul_param_size %d is remove %d\n",
        //         params_list[i].pointer, _param.type, _param.size, _param.index, total_eff_state_dim, _param.is_remove);
        if (_param.is_remove) {
            remove_state_dim += _param.eff_size;
        } else {
            keep_block_size ++;
        }
        params_list[i] = _param;
    }
}

void ResidualInfo::Evaluate(std::vector<double*> params) {
    //This function is from VINS.
    residuals.resize(cost_function->num_residuals());
    std::vector<int> blk_sizes = cost_function->parameter_block_sizes();
    double ** raw_jacobians = new double *[blk_sizes.size()];
    jacobians.resize(blk_sizes.size());
    for (int i = 0; i < static_cast<int>(blk_sizes.size()); i++) {
        jacobians[i].resize(cost_function->num_residuals(), blk_sizes[i]);
        jacobians[i].setZero();
        raw_jacobians[i] = jacobians[i].data();
    }
    cost_function->Evaluate(params.data(), residuals.data(), raw_jacobians);
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(params.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

void ImuResInfo::Evaluate(D2EstimatorState * state) {
    std::vector<double*> params{state->getPoseState(frame_ida), state->getSpdBiasState(frame_ida), 
        state->getPoseState(frame_idb), state->getSpdBiasState(frame_idb)};
    ((ResidualInfo*)this)->Evaluate(params);
}

void LandmarkTwoFrameOneCamResInfo::Evaluate(D2EstimatorState * state) {
    std::vector<double*> params{state->getPoseState(frame_ida), 
                    state->getPoseState(frame_idb), 
                    state->getExtrinsicState(camera_id),
                    state->getLandmarkState(landmark_id)};
    ((ResidualInfo*)this)->Evaluate(params);
}

void LandmarkTwoFrameOneCamResInfoTD::Evaluate(D2EstimatorState * state) {
    std::vector<double*> params{state->getPoseState(frame_ida), 
                    state->getPoseState(frame_idb), 
                    state->getExtrinsicState(camera_id),
                    state->getLandmarkState(landmark_id),
                    state->getTdState(camera_id)};
    ((ResidualInfo*)this)->Evaluate(params);
}

PriorResInfo::PriorResInfo(PriorFactor * _factor)
    :ResidualInfo(PriorResidual) {
    cost_function = _factor;
    factor = _factor;
}

bool PriorResInfo::relavant(const std::set<FrameIdType> & frame_ids) const {
    //Prior relavant to all frames.
    return true;
}

void PriorResInfo::Evaluate(D2EstimatorState * state) {
    std::vector<double*> params = factor->getKeepParamsPointers();
    ((ResidualInfo*)this)->Evaluate(params);
}

std::vector<ParamInfo> PriorResInfo::paramsList(D2EstimatorState * state) const {
    return factor->getKeepParams();
}

}