#include <d2common/solver/consenus_factor.h>

#include <d2common/utils.hpp>

namespace D2Common {
ConsenusPoseFactor::ConsenusPoseFactor(Eigen::Vector3d _t_ref,
                                       Eigen::Quaterniond _q_ref,
                                       Eigen::Vector3d _t_tilde,
                                       Eigen::Vector3d _theta_tilde,
                                       double rho_T, double rho_theta)
    : t_ref(_t_ref),
      q_ref(_q_ref),
      t_tilde(_t_tilde),
      theta_tilde(_theta_tilde) {
  q_sqrt_info = Eigen::Matrix3d::Identity() * rho_T;
  T_sqrt_info = Eigen::Matrix3d::Identity() * rho_theta;
}

bool ConsenusPoseFactor::Evaluate(double const *const *parameters,
                                  double *residuals, double **jacobians) const {
  Eigen::Map<const Eigen::Vector3d> T_local(parameters[0]);
  Eigen::Map<const Eigen::Quaterniond> q_local(parameters[0] + 3);
  Eigen::Map<Eigen::Vector3d> T_err(residuals);
  Eigen::Map<Eigen::Vector3d> theta_err(residuals + 3);

  Matrix3d R_ref_inv = q_ref.toRotationMatrix().transpose();

  Eigen::Quaterniond q_err = q_ref.inverse() * q_local;
  theta_err = 2.0 * q_err.vec();
  theta_err = q_sqrt_info * (theta_err + theta_tilde);
  T_err = T_sqrt_info * (R_ref_inv * (T_local - t_ref) + t_tilde);
  if (T_err.hasNaN()) {
    std::cout << "q_ref" << q_ref.coeffs().transpose() << std::endl;
    std::cout << "t_ref" << t_ref.transpose() << std::endl;
    std::cout << "T_local" << T_local.transpose() << std::endl;
    std::cout << "t_tilde" << t_tilde.transpose() << std::endl;
    std::cout << "T_err" << T_err.transpose() << std::endl;
    std::cout << "T_sqrt_info\n" << T_sqrt_info << std::endl;
    std::cout << "R_ref_inv\n" << R_ref_inv << std::endl;
  }
  if (jacobians) {
    // Fill in jacobians...
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>>
        jacobian_pose_local(jacobians[0]);
    jacobian_pose_local.setZero();
    jacobian_pose_local.block<3, 3>(0, 0) = T_sqrt_info * R_ref_inv;
    jacobian_pose_local.block<3, 3>(3, 3) =
        q_sqrt_info * Utility::Qleft(q_err).bottomRightCorner<3, 3>();
  }
  return true;
}

}  // namespace D2Common