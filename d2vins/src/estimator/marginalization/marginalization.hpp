#pragma once
#include <d2common/d2vinsframe.h>
#include <ceres/ceres.h>
#include "marginalization_info.hpp"

namespace D2VINS {
class Marginalizer {
protected:
    D2EstimatorState * state = nullptr;
    std::vector<ResidualInfo*> residual_info_list;

    //To remove ids
    std::set<FrameIdType> remove_frame_ids;

    //Interal params
    //sorted params
    std::vector<ParamInfo> params_list; //[parameters... remove_params...] true if remove
    std::map<state_type*, ParamInfo> _params; // indx of parameters in params vector as sortd by params_list

    void sortParams();
    VectorXd evaluate(SparseMat & J, int eff_residual_size, int eff_param_size);
    void covarianceEstimation(const SparseMat & H);
    int filterResiduals();
    int remove_state_dim = 0;
    int total_eff_state_dim = 0;
    int keep_block_size = 0;
public:
    Marginalizer(D2EstimatorState * _state): state(_state) {}
    void addLandmarkResidual(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id, bool has_td=false);
    void addDepthResidual(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, LandmarkIdType landmark_id);
    void addLandmarkResidualOneFrameTwoCam(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, LandmarkIdType landmark_id, int camera_id_a, int camera_id_b);
    void addLandmarkResidualTwoFrameTwoCam(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id_a, int camera_id_b);
    void addLandmarkResidualTwoFrameTwoCamDistrib(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id_a, int camera_id_b);
    void addImuResidual(ceres::CostFunction * cost_function, FrameIdType frame_ida, FrameIdType frame_idb);
    void addPrior(PriorFactor * cost_function);
    PriorFactor * marginalize(std::set<FrameIdType> remove_frame_ids);
};
}