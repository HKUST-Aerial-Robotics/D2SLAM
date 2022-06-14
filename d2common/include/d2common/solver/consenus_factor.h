#pragma once
#include <ceres/ceres.h>

namespace D2Common {
class ConsenusPoseFactor : public ceres::SizedCostFunction<6, 7> {
    Eigen::Matrix<double, 3, 3> q_sqrt_info;
    Eigen::Matrix<double, 3, 3> T_sqrt_info;
    Eigen::Vector3d t_ref;
    Eigen::Quaterniond q_ref;

    Eigen::Vector3d t_tilde;
    Eigen::Vector3d theta_tilde;
public:
    ConsenusPoseFactor(Eigen::Vector3d _t_ref, Eigen::Quaterniond _q_ref, 
            Eigen::Vector3d _t_tilde, Eigen::Vector3d _theta_tilde, double rho_T, double rho_theta);

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
};
}