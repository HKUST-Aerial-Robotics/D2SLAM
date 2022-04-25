#include "prior_factor.h"
#include "../estimator/marginalize.hpp"

namespace D2VINS {

PriorFactor::PriorFactor(std::vector<ParamInfo> _keep_params_list): 
    keep_params_list(_keep_params_list)
{
    keep_param_size = keep_params_list.size();
}

bool PriorFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::VectorXd dx(keep_param_size);
    for (int i = 0; i < keep_param_size; i++)
    {
        auto & info = keep_params_list[i];
        int size = info.size;
        int idx = info.index;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(info.pointer, size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, keep_param_size) = linearized_res + linearized_jac * dx;
    if (jacobians)
    {

        for (int i = 0; i < keep_param_size; i++)
        {
            if (jacobians[i])
            {
                auto & info = keep_params_list[i];
                int size = info.size;
                int idx = info.index;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], keep_param_size, size);
                jacobian.setZero();
                auto local_size = size == 7 ? 6 : size;
                jacobian.leftCols(local_size) = linearized_jac.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
}