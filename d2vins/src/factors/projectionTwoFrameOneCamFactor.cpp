/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science
 *and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "projectionTwoFrameOneCamFactor.h"

#include <d2common/utils.hpp>

#include "../d2vins_params.hpp"
using namespace D2Common;

namespace D2VINS {
Eigen::Matrix2d ProjectionTwoFrameOneCamFactor::sqrt_info;
double ProjectionTwoFrameOneCamFactor::sum_t;

ProjectionTwoFrameOneCamFactor::ProjectionTwoFrameOneCamFactor(
    const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
    const Eigen::Vector3d &_velocity_i, const Eigen::Vector3d &_velocity_j,
    const double _td_i, const double _td_j)
    : pts_i(_pts_i),
      pts_j(_pts_j),
      td_i(_td_i),
      td_j(_td_j),
      velocity_i(_velocity_i),
      velocity_j(_velocity_j) {
#ifdef UNIT_SPHERE_ERROR
  Eigen::Vector3d b1, b2;
  Eigen::Vector3d a = pts_j.normalized();
  Eigen::Vector3d tmp(0, 0, 1);
  if (a == tmp) tmp << 1, 0, 0;
  b1 = (tmp - a * (a.transpose() * tmp)).normalized();
  b2 = a.cross(b1);
  tangent_base.block<1, 3>(0, 0) = b1.transpose();
  tangent_base.block<1, 3>(1, 0) = b2.transpose();
  // printf("vel i %.4f %.4f %.4f  j %.4f %.4f %.4f\n", velocity_i(0),
  // velocity_i(1), velocity_i(2), velocity_j(0), velocity_j(1), velocity_j(2));
#endif
};

bool ProjectionTwoFrameOneCamFactor::Evaluate(double const *const *parameters,
                                              double *residuals,
                                              double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4],
                        parameters[0][5]);

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4],
                        parameters[1][5]);

  Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4],
                         parameters[2][5]);

  double inv_dep_i = parameters[3][0];

  double td = parameters[4][0];

  Eigen::Vector3d pts_i_td, pts_j_td;
  pts_i_td = pts_i - (td - td_i) * velocity_i;
  pts_j_td = pts_j - (td - td_j) * velocity_j;
  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
  Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR
  residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif

  residual = sqrt_info * residual;

  if (jacobians) {
    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
    const auto ric_t = ric.transpose();
    const auto Rj_t = Rj.transpose();
    const auto J_w = ric_t * Rj_t;
    const auto J_imu_i = J_w * Ri;
    const auto J_cam_i = J_imu_i * ric;

#ifdef UNIT_SPHERE_ERROR
    double norm = pts_camera_j.norm();
    Eigen::Matrix3d norm_jaco;
    double x1, x2, x3;
    x1 = pts_camera_j(0);
    x2 = pts_camera_j(1);
    x3 = pts_camera_j(2);
    double norm_3 = pow(norm, 3);
    norm_jaco << 1.0 / norm - x1 * x1 / norm_3, -x1 * x2 / norm_3,
        -x1 * x3 / norm_3, -x1 * x2 / norm_3, 1.0 / norm - x2 * x2 / norm_3,
        -x2 * x3 / norm_3, -x1 * x3 / norm_3, -x2 * x3 / norm_3,
        1.0 / norm - x3 * x3 / norm_3;
    reduce = tangent_base * norm_jaco;
    Matrix3d reduce_j_td;
    x1 = pts_j_td(0);
    x2 = pts_j_td(1);
    x3 = pts_j_td(2);
    norm_3 = pow(pts_j_td.norm(), 3);
    reduce_j_td << 1.0 / norm - x1 * x1 / norm_3, -x1 * x2 / norm_3,
        -x1 * x3 / norm_3, -x1 * x2 / norm_3, 1.0 / norm - x2 * x2 / norm_3,
        -x2 * x3 / norm_3, -x1 * x3 / norm_3, -x2 * x3 / norm_3,
        1.0 / norm - x3 * x3 / norm_3;
#else
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j), 0, 1. / dep_j,
        -pts_camera_j(1) / (dep_j * dep_j);
#endif
    reduce = sqrt_info * reduce;

    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(
          jacobians[0]);

      Eigen::Matrix<double, 3, 6> jaco_i;
      jaco_i.leftCols<3>() = J_w;
      jaco_i.rightCols<3>() = J_imu_i * -Utility::skewSymmetric(pts_imu_i);

      jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
      jacobian_pose_i.rightCols<1>().setZero();
    }

    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(
          jacobians[1]);

      Eigen::Matrix<double, 3, 6> jaco_j;
      jaco_j.leftCols<3>() = -J_w;
      jaco_j.rightCols<3>() = ric_t * Utility::skewSymmetric(pts_imu_j);

      jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
      jacobian_pose_j.rightCols<1>().setZero();
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(
          jacobians[2]);
      Eigen::Matrix<double, 3, 6> jaco_ex;
      jaco_ex.leftCols<3>() = J_imu_i - ric_t;
      jaco_ex.rightCols<3>() =
          -J_cam_i * Utility::skewSymmetric(pts_camera_i) +
          Utility::skewSymmetric(J_cam_i * pts_camera_i) +
          Utility::skewSymmetric(J_w * (Ri * tic + Pi - Pj) - Rj_t * tic);
      jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
      jacobian_ex_pose.rightCols<1>().setZero();
    }
    if (jacobians[3]) {
      Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
      jacobian_feature =
          reduce * J_cam_i * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
    }
    if (jacobians[4]) {
      Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);
#ifdef UNIT_SPHERE_ERROR
      jacobian_td = reduce * J_cam_i * velocity_i / inv_dep_i * -1.0 +
                    sqrt_info * tangent_base * reduce_j_td * velocity_j;
#else
      jacobian_td = reduce * J_cam_i * velocity_i / inv_dep_i * -1.0 +
                    sqrt_info * velocity_j.head(2);
#endif
    }
  }
  return true;
}

void ProjectionTwoFrameOneCamFactor::check(double **parameters) {
  double *res = new double[2];
  double **jaco = new double *[5];
  jaco[0] = new double[2 * 7];
  jaco[1] = new double[2 * 7];
  jaco[2] = new double[2 * 7];
  jaco[3] = new double[2 * 1];
  jaco[4] = new double[2 * 1];
  Evaluate(parameters, res, jaco);
  puts("check begins");

  puts("my");

  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose()
            << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0])
            << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[1])
            << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[2])
            << std::endl
            << std::endl;
  std::cout << Eigen::Map<Eigen::Vector2d>(jaco[3]) << std::endl << std::endl;
  std::cout << Eigen::Map<Eigen::Vector2d>(jaco[4]) << std::endl << std::endl;

  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4],
                        parameters[0][5]);

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4],
                        parameters[1][5]);

  Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4],
                         parameters[2][5]);
  double inv_dep_i = parameters[3][0];
  double td = parameters[4][0];

  Eigen::Vector3d pts_i_td, pts_j_td;
  pts_i_td = pts_i - (td - td_i) * velocity_i;
  pts_j_td = pts_j - (td - td_j) * velocity_j;
  Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
  Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
  Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
  Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
  Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
  Eigen::Vector2d residual;

#ifdef UNIT_SPHERE_ERROR
  residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
  double dep_j = pts_camera_j.z();
  residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
  residual = sqrt_info * residual;

  puts("num");
  std::cout << residual.transpose() << std::endl;

  const double eps = 1e-6;
  Eigen::Matrix<double, 2, 20> num_jacobian;
  for (int k = 0; k < 20; k++) {
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4],
                          parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4],
                          parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4],
                           parameters[2][5]);
    double inv_dep_i = parameters[3][0];
    double td = parameters[4][0];

    int a = k / 3, b = k % 3;
    Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

    if (a == 0)
      Pi += delta;
    else if (a == 1)
      Qi = Qi * Utility::deltaQ(delta);
    else if (a == 2)
      Pj += delta;
    else if (a == 3)
      Qj = Qj * Utility::deltaQ(delta);
    else if (a == 4)
      tic += delta;
    else if (a == 5)
      qic = qic * Utility::deltaQ(delta);
    else if (a == 6 && b == 0)
      inv_dep_i += delta.x();
    else if (a == 6 && b == 1)
      td += delta.y();

    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Vector2d tmp_residual;

#ifdef UNIT_SPHERE_ERROR
    tmp_residual =
        tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
    tmp_residual = sqrt_info * tmp_residual;

    num_jacobian.col(k) = (tmp_residual - residual) / eps;
  }
  std::cout << num_jacobian.block<2, 6>(0, 0) << std::endl;
  std::cout << num_jacobian.block<2, 6>(0, 6) << std::endl;
  std::cout << num_jacobian.block<2, 6>(0, 12) << std::endl;
  std::cout << num_jacobian.block<2, 1>(0, 18) << std::endl;
  std::cout << num_jacobian.block<2, 1>(0, 19) << std::endl;
}
}  // namespace D2VINS