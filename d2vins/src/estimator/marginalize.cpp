#include "marginalize.hpp"
#include <d2vins/utils.hpp>

namespace D2VINS {
void Marginalizer::addLandmarkResidual(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
    FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id) {
    auto * info = new LandmarkTwoFrameOneCamResInfo();
    info->frame_ida = frame_ida;
    info->frame_idb = frame_idb;
    info->landmark_id = landmark_id;
    info->camera_id = camera_id;
    info->cost_function = cost_function;
    info->loss_function = loss_function;
    info->parameter_size = 3*POSE_SIZE + (params->landmark_param == D2VINSConfig::LM_INV_DEP ? INV_DEP_SIZE : POS_SIZE);
    residual_info_list.push_back(info);
}

void Marginalizer::addImuResidual(ceres::CostFunction * cost_function,  FrameIdType frame_ida, FrameIdType frame_idb) {
    auto * info = new ImuResInfo();
    info->frame_ida = frame_ida;
    info->frame_idb = frame_idb;
    info->cost_function = cost_function;
    info->loss_function = nullptr;
    info->parameter_size = 2*POSE_SIZE + 2*FRAME_SPDBIAS_SIZE;
    residual_info_list.push_back(info);
}

void Marginalizer::addFramePoseParams(FrameIdType frame_id) {
    auto _ptr = state->getPoseState(frame_id);
    auto _ptr_spd_bias = state->getSpdBiasState(frame_id);
    bool is_remove = false;
    if (remove_frame_ids.find(frame_id) != remove_frame_ids.end()) {
        is_remove = true;
    }
    addParam(_ptr, POSE, frame_id, is_remove);
    addParam(_ptr_spd_bias, SPEED_BIAS, frame_id, is_remove);
}

void Marginalizer::addLandmarkStateParam(LandmarkIdType landmark_id) {
    auto _ptr = state->getLandmarkState(landmark_id);
    FrameIdType base_frame_id = state->getLandmarkBaseFrame(landmark_id);
    bool is_remove = false;
    if (remove_frame_ids.find(base_frame_id) != remove_frame_ids.end()) {
        is_remove = true;
        // printf("Landmark %ld inv_dep will be removed\n", landmark_id);
    }
    addParam(_ptr, LANDMARK, landmark_id, is_remove);
}

void Marginalizer::addExtrinsicParam(int camera_id) {
    auto _ptr = state->getExtrinsicState(camera_id);
    addParam(_ptr, EXTRINSIC, camera_id, false);
}

void Marginalizer::addParam(state_type * pointer, ParamsType type, FrameIdType _id, bool is_remove) {
    if (_params.find(pointer) == _params.end()) {
        // printf("Add param block %p\n", pointer);
    } else {
        return;
    }

    ParamInfo param;
    param.pointer = pointer;
    param.type = type;
    param.id = _id;
    param.is_remove = is_remove;
    if (type == SPEED_BIAS) {
        param.size = FRAME_SPDBIAS_SIZE;
    } else if (type == LANDMARK) {
        if (params->landmark_param == D2VINSConfig::LM_INV_DEP) {
            param.size = INV_DEP_SIZE;
        } else {
            param.size = POS_SIZE;
        }
    }
    _params[pointer] = param;
}


void Marginalizer::marginalize(std::set<FrameIdType> _remove_frame_ids) {
    TicToc tic;
    remove_frame_ids = _remove_frame_ids;
    //Clear previous states
    params_list.clear();
    _params.clear();
    //We first remove all factors that does not evolved frame
    int eff_residual_size = 0;
    for (auto it = residual_info_list.begin(); it != residual_info_list.end();) {
        if ((*it)->relavant(remove_frame_ids)) {
            eff_residual_size += (*it)->residualSize();
            if ((*it)->residual_type == ResidualType::IMUResidual) {
                auto info = static_cast<ImuResInfo*>(*it);
                addFramePoseParams(info->frame_ida);
                addFramePoseParams(info->frame_idb);
                // printf("Adding residual of IMU %ld<->%ld\n", info->frame_ida, info->frame_idb);
            } else if ((*it)->residual_type == ResidualType::LandmarkTwoFrameOneCamResidual) {
                auto info = static_cast<LandmarkTwoFrameOneCamResInfo*>(*it);
                // printf("Adding landmark of %ld<->%ld\n", info->frame_ida, info->frame_idb);
                addFramePoseParams(info->frame_ida);
                addFramePoseParams(info->frame_idb);
                addExtrinsicParam(info->camera_id);
                addLandmarkStateParam(info->landmark_id);
            } else {
                printf("Unknown residual type\n");
            }
            it++;
        } else {
            delete *it;
            it = residual_info_list.erase(it);
        }
    }
    auto ret = sortParams(); //sort the parameters
    auto eff_param_size = ret.first;
    auto remove_state_size = ret.second;
    int keep_state_size = eff_param_size - remove_state_size;
    printf("[D2VINS::Marginalizer::marginalize] frame_id %ld eff_param_size: %d remove param size %d eff_residual_size: %d \n", 
         *remove_frame_ids.begin(), eff_param_size, remove_state_size, eff_residual_size);

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
            auto j0 = _params.at(params[param_blk_i].first).index;
            auto param_size = params[param_blk_i].second;
            for (auto i = 0; i < residual_size; i ++) {
                for (auto j = 0; j < param_size; j ++) {
                    triplet_list.push_back(Eigen::Triplet<state_type>(i0 + i, j0 + j, J_blk(i, j)));
                    // printf("J[%d, %d] = %f\n", i0 + i, j0 + j, J_blk(i, j));
                    if (i0 + i >= eff_residual_size || j0 + j >= eff_param_size) {
                        printf("J[%d, %d] = %f\n", i0 + i, j0 + j, J_blk(i, j));
                        fflush(stdout);
                        exit(0);
                    }
                }
            }
        }
        cul_res_size += residual_size;
    }

    TicToc tic_j;
    SparseMat J(eff_residual_size, eff_param_size);
    J.setFromTriplets(triplet_list.begin(), triplet_list.end());
    SparseMat H = SparseMatrix<double>(J.transpose())*J;
    // printf("[D2VINS::Marginalizer] J^TJ time %.3fms\n", tic_j.toc());

    tic_j.tic();
    SparseMat H11 = H.block(0, 0, keep_state_size, keep_state_size);
    SparseMat H12 = H.block(0, keep_state_size, keep_state_size, remove_state_size);
    SparseMat H22 = H.block(keep_state_size, keep_state_size, remove_state_size, remove_state_size);
    SparseMat H22_inv = Utility::inverse(H22);
    SparseMat A = H11 - H12 * H22_inv * SparseMatrix<double>(H12.transpose());
    VectorXd b = residual_vec.segment(0, keep_state_size) - H12 * H22_inv * residual_vec.segment(keep_state_size, remove_state_size);
    printf("[D2VINS::Marginalizer] time cost %.1fms\n", tic.toc());
    // tic_j.tic();
    auto linearized_jac = Utility::Linv(A, b);
    // Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    // solver.compute(A);
    // // b = solver.permutationP() * b;
    // auto L = solver.derived().matrixL();
    // std::cout << L << std::endl;
    // auto linear_jac = L.toDense();
    // std::cout << linear_jac << std::endl;
    // printf("b rows %d cols %d L rows %d cols %d\n", b.rows(), b.cols(), L.rows(), L.cols());
    // std::cout << L.toDense() << std::endl;
    // L.toDense().solveInPlace(b);
    // printf("[D2VINS::Marginalizer] linearized_jac time cost %.3fms\n", tic_j.toc());
}

std::pair<int, int> Marginalizer::sortParams() {
    int remove_size = 0;
    int total_size = 0;
    params_list.clear();
    for (auto it: _params) {
        params_list.push_back(it.second);
    }

    std::sort(params_list.begin(), params_list.end(), [](const ParamInfo & a, const ParamInfo & b) {
        return a.is_remove < b.is_remove;
    });

    int cul_param_size = 0;
    for (unsigned i = 0; i < params_list.size(); i++) {
        auto & _param = _params.at(params_list[i].pointer);
        _param.index = cul_param_size;
        cul_param_size += _param.size;
        if (_param.is_remove) {
            remove_size += _param.size;
        }
        total_size += _param.size;
    }
    return make_pair(total_size, remove_size);
}

void ResidualInfo::Evaluate(std::vector<double*> params) {
    //This function is from VINS.
    residuals.resize(cost_function->num_residuals());
    std::vector<int> blk_sizes = cost_function->parameter_block_sizes();
    double ** raw_jacobians = new double *[blk_sizes.size()];
    jacobians.resize(blk_sizes.size());
    for (int i = 0; i < static_cast<int>(blk_sizes.size()); i++) {
        jacobians[i].resize(cost_function->num_residuals(), blk_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
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


}
