#include "marginalization.hpp"
#include <d2common/utils.hpp>
#include "../../factors/prior_factor.h"
#include "../../factors/imu_factor.h"
#include "../../factors/projectionTwoFrameOneCamFactor.h"

using namespace D2Common;

namespace D2VINS {
void Marginalizer::addResidualInfo(ResidualInfo* info) {
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
            if (std::isnan(J_blk.maxCoeff()) || std::isnan(J_blk.minCoeff())) {
                printf("\033[0;31m[D2VINS::Marginalizer] Residual type %d param_blk %d jacobians is nan\033[0m:\n",
                    info->residual_type, param_blk_i);
                std::cout << J_blk << std::endl;
            }
            if (std::isnan(info->residuals.maxCoeff()) || std::isnan(info->residuals.minCoeff())) {
                printf("\033[0;31m[D2VINS::Marginalizer] Residual type %d param_blk %d residuals is nan\033[0m\n", 
                    info->residual_type, param_blk_i);
                std::cout << info->residuals.transpose() << std::endl;
            }

            for (auto i = 0; i < residual_size; i ++) {
                for (auto j = 0; j < blk_eff_param_size; j ++) {
                    //We only copy the eff param part, that is: on tangent space.
                    triplet_list.push_back(Eigen::Triplet<state_type>(i0 + i, j0 + j, J_blk(i, j)));
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
            for (auto param_ : param_list) {
                if (_params.find(param_.pointer) == _params.end()) {
                    _params[param_.pointer] = param_;
                    //Check if param should be remove
                    bool is_remove = false;
                    auto & param = _params[param_.pointer];
                    if ((param.type == POSE || param.type == SPEED_BIAS) && remove_frame_ids.find(param.id) != remove_frame_ids.end()) {
                        param.is_remove = true;
                    }
                    if (param.type == LANDMARK) {
                        FrameIdType base_frame_id = state->getLandmarkBaseFrame(param.id);
                        param.is_remove = false;
                        if (remove_frame_ids.find(base_frame_id) != remove_frame_ids.end()) {
                            param.is_remove = true;
                        } else {
                            if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
                                // printf("[D2VINS::Marginalizer::filterResiduals] landmark %d base frame %d not in remove_frame_ids %ld but will be remove\n", param.id, base_frame_id, *remove_frame_ids.begin());
                                param.is_remove = params->remove_base_when_margin_remote;
                            }
                        }
                    }
                }
            }
            it++;
        } else {
            delete *it;
            it = residual_info_list.erase(it);
        }
    }
    return eff_residual_size;
}

void Marginalizer::covarianceEstimation(const SparseMat & H) {
    //This covariance estimation is for debug only. It does not estimate the real covariance in marginalization module. 
    //To make it estimate real cov, we need to use full jacobian instead part of the problem.
    auto cov = Utility::inverse(H);
    printf("covarianceEstimation\n");
    for (auto param: params_list) {
        if (param.type == POSE) {
            //Print pose covariance
            printf("[D2VINS::Marginalizer::covarianceEstimation] pose %d covariance:\n", param.id);
            std::cout << cov.block(param.index, param.index, POSE_EFF_SIZE, POSE_EFF_SIZE) << std::endl;
        }
        if (param.type == EXTRINSIC) {
            //Print extrinsic covariance
            printf("[D2VINS::Marginalizer::covarianceEstimation] extrinsic %d covariance:\n", param.id);
            std::cout << cov.block(param.index, param.index, POSE_EFF_SIZE, POSE_EFF_SIZE) << std::endl;
        }
        if (param.type == SPEED_BIAS) {
            //Print speed bias covariance
            printf("[D2VINS::Marginalizer::covarianceEstimation] speed bias %d covariance:\n", param.id);
            std::cout << cov.block(param.index, param.index, FRAME_SPDBIAS_SIZE, FRAME_SPDBIAS_SIZE) << std::endl;
        }
    }
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
    TicToc tt;
    SparseMat J(eff_residual_size, total_eff_state_dim);
    auto b = evaluate(J, eff_residual_size, total_eff_state_dim);
    SparseMat H = SparseMatrix<double>(J.transpose())*J;
    VectorXd g = J.transpose()*b; //Ignore -b here and also in prior_factor.cpp toJacRes to reduce compuation
    if (params->enable_perf_output) {
        printf("[D2VINS::Marginalizer::marginalize] JtJ cost %.1fms\n", tt.toc());
    }
    if (keep_block_size == 0 || remove_state_dim == 0) {
        printf("\033[0;31m[D2VINS::Marginalizer::marginalize] keep_block_size=%d remove_state_dim%d\033[0m\n", keep_block_size, remove_state_dim);
        return nullptr;
    }
    std::vector<ParamInfo> keep_params_list(params_list.begin(), params_list.begin() + keep_block_size);
    //Compute the schur complement, by sparse LLT.
    PriorFactor * prior = nullptr;
    if (params->margin_sparse_solver) {
        tt.tic();
        auto Ab = Utility::schurComplement(H, g, keep_state_dim);
        if (params->enable_perf_output) {
            printf("[D2VINS::Marginalizer::marginalize] schurComplement cost %.1fms\n", tt.toc());
        }
        prior = new PriorFactor(keep_params_list, Ab.first, Ab.second);
    } else {
        auto Ab = Utility::schurComplement(H.toDense(), g, keep_state_dim);
        prior = new PriorFactor(keep_params_list, Ab.first, Ab.second);
    }

    if (params->enable_perf_output || params->verbose) {
        printf("[D2VINS::Marginalizer::marginalize] time cost %.1fms frame_id %ld total_eff_state_dim: %d keep_size %d remove size %d eff_residual_size: %d keep_block_size %d \n", 
            tic.toc(), *remove_frame_ids.begin(), total_eff_state_dim, keep_state_dim, remove_state_dim, eff_residual_size, keep_block_size);
    }

    if (params->debug_write_margin_matrix) {
        Utility::writeMatrixtoFile(params->output_folder + "/H.txt", MatrixXd(H));
        Utility::writeMatrixtoFile(params->output_folder + "/g.txt", MatrixXd(g));
    }
    // for (auto & param : keep_params_list) {
    //     printf("param id %d type %d size %d \n", param.id, param.type, param.size);
    //     std::cout << "H:\n" << H.block(param.index, 0, param.eff_size, total_eff_state_dim) << std::endl;
    //     std::cout << "g:" << g.segment(param.index, param.eff_size).transpose() << std::endl;
    // }
    if (prior->hasNan()) {
        printf("\033[0;31m[D2VINS::Marginalizer::marginalize] prior has nan\033[0m\n");
        return nullptr;
    }
    return prior;
}

void Marginalizer::sortParams() {
    params_list.clear();
    for (auto it: _params) {
        params_list.push_back(it.second);
    }

    std::sort(params_list.begin(), params_list.end(), [](const ParamInfo & a, const ParamInfo & b) {
        if (a.is_remove != b.is_remove) {
            return a.is_remove < b.is_remove;
        } else {
            return a.type < b.type;
        }
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


}