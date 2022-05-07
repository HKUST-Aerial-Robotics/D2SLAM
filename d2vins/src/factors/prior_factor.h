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
            // std::cout << "linearized_jac max " << linearized_jac.maxCoeff() << std::endl;
            // std::cout << "linearized_res max " << linearized_res.maxCoeff() << std::endl;
        }
        // std::cout << "A\n" << A.block(0, 0, 7, 7) << std::endl;
        // std::cout << "b " << b.segment(0, 7).transpose() << std::endl;
        // std::cout << "linearized_jac\n" << linearized_jac.block(0, 0, 7, 7) << std::endl;
        // std::cout << "linearized_res\n" << linearized_res.segment(0, 7).transpose() << std::endl;
        initDims(_keep_params_list);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    virtual std::vector<state_type *> getKeepParamsPointers() const;
    virtual std::vector<ParamInfo> getKeepParams() const;
    int getEffParamsDim() const;
    bool hasNan() const {
        if (std::isnan(linearized_jac.maxCoeff()) || std::isnan(linearized_res.minCoeff())) {
            printf("\033[0;31m [D2VINS::PriorFactor] linearized_jac has NaN\033[0m\n");
            // std::cout << "linearized_jac\n" <<  linearized_jac << std::endl;
            return true;
        }
        if (std::isnan(linearized_res.maxCoeff()) || std::isnan(linearized_res.minCoeff())) {
            printf("\033[0;31m [D2VINS::PriorFactor] linearized_res has NaN\033[0m\n");
            // std::cout << "linearized_res\n" <<  linearized_res << std::endl;
            return true;
        }
        return false;
    }

};

}