#include "marginalization_info.hpp"
#include "../../factors/prior_factor.h"

namespace D2VINS {
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
                    state->getExtrinsicState(camera_index),
                    state->getLandmarkState(landmark_id)};
    ((ResidualInfo*)this)->Evaluate(params);
}

void LandmarkTwoFrameOneCamResInfoTD::Evaluate(D2EstimatorState * state) {
    std::vector<double*> params{state->getPoseState(frame_ida), 
                    state->getPoseState(frame_idb), 
                    state->getExtrinsicState(camera_index),
                    state->getLandmarkState(landmark_id),
                    state->getTdState(camera_index)};
    ((ResidualInfo*)this)->Evaluate(params);
}

void LandmarkTwoFrameTwoCamResInfoTD::Evaluate(D2EstimatorState * state) {
    std::vector<double*> params{state->getPoseState(frame_ida), 
                    state->getPoseState(frame_idb), 
                    state->getExtrinsicState(camera_index_a),
                    state->getExtrinsicState(camera_index_b),
                    state->getLandmarkState(landmark_id),
                    state->getTdState(camera_index_a)};
    ((ResidualInfo*)this)->Evaluate(params);
}

void LandmarkOneFrameTwoCamResInfoTD::Evaluate(D2EstimatorState * state) {
    std::vector<double*> params{state->getExtrinsicState(camera_index_a),
                    state->getExtrinsicState(camera_index_b),
                    state->getLandmarkState(landmark_id),
                    state->getTdState(camera_index_a)};
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