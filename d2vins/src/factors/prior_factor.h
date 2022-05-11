#pragma once
#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <d2vins/d2vins_types.hpp>

// This is devied from VINS-Mono
namespace D2VINS
{
struct ParamInfo;

std::pair<MatrixXd, VectorXd> toJacRes(const SparseMat & A, const VectorXd & b);
std::pair<MatrixXd, VectorXd> toJacRes(const MatrixXd & A, const VectorXd & b);

class PriorFactor : public ceres::CostFunction {
    std::vector<ParamInfo> keep_params_list;
    int keep_param_blk_num = 0;
    Eigen::MatrixXd linearized_jac;
    Eigen::VectorXd linearized_res;
    int keep_eff_param_dim = -1;
    void initDims(const std::vector<ParamInfo> & _keep_params_list);
public:
    template <typename MatrixType>
    PriorFactor(const std::vector<ParamInfo> & _keep_params_list, const MatrixType &A, const VectorXd &b) {
        TicToc tic_j;
        auto ret = toJacRes(A, b);
        linearized_jac = ret.first;
        linearized_res = ret.second;

        if (hasNan()) {
            std::cout << "NaN found in Prior factor" << std::endl;
            std::cout << "A max " << MatrixXd(A).maxCoeff() << std::endl;
            std::cout << "b max " << b.maxCoeff() << std::endl;
        }
        initDims(_keep_params_list);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    virtual std::vector<state_type *> getKeepParamsPointers() const;
    virtual std::vector<ParamInfo> getKeepParams() const;
    int getEffParamsDim() const;
    bool hasNan() const;
};

}