#pragma once
#include <d2common/d2vinsframe.h>
#include <ceres/ceres.h>
#include "../ParamResidualInfo.hpp"

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
    std::map<StatePtr, ParamInfo> _params; // indx of parameters in params vector as sortd by params_list

    int remove_state_dim = 0;
    int total_eff_state_dim = 0;
    int keep_block_size = 0;
    PriorFactor * last_prior = nullptr;

    void sortParams();
    VectorXd evaluate(SparseMat & J, int eff_residual_size, int eff_param_size);
    void covarianceEstimation(const SparseMat & H);
    int filterResiduals();
    void showDeltaXofschurComplement(std::vector<ParamInfo> keep_params_list, const SparseMatrix<double> & A, const Matrix<double, Dynamic, 1> & b);
public:
    Marginalizer(D2EstimatorState * _state, PriorFactor*_last): state(_state), last_prior(_last) {}
    void addResidualInfo(ResidualInfo* info);
    void addPrior(PriorFactor * cost_function);
    PriorFactor * marginalize(std::set<FrameIdType> remove_frame_ids);
};
}