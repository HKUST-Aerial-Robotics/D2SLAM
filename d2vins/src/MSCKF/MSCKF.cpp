#include "MSCKF.hpp"

#include <d2common/utils.hpp>

namespace D2VINS {
MSCKF::MSCKF() {
  Q_imu.setZero();
  Q_imu.block<3, 3>(0, 0) = _config.gyr_n * Matrix3d::Identity();
  Q_imu.block<3, 3>(3, 3) = _config.gyr_w * Matrix3d::Identity();
  Q_imu.block<3, 3>(6, 6) = _config.acc_n * Matrix3d::Identity();
  Q_imu.block<3, 3>(9, 9) = _config.acc_w * Matrix3d::Identity();
}

void MSCKF::initFirstPose() {
  auto q0 = Utility::g2R(imubuf.mean_acc());
  nominal_state.q_imu = q0;
}

void MSCKF::predict(const double t, const IMUData& imudata) {
  // Follows  Mourikis, Anastasios I., and Stergios I. Roumeliotis.
  // "A multi-state constraint Kalman filter for vision-aided inertial
  // navigation." Proceedings 2007 IEEE International Conference on Robotics and
  // Automation. IEEE, 2007. Sect III-B
  if (t_last < 0) {
    t_last = t;
  }

  imubuf.add(imudata);
  if (!initFirstPoseFlag) {
    // First pose is inited in keyframe. So first pose must is a keyframe.
    return;
  }

  double dt = t - t_last;

  static IMUData imudatalast;

  // trapezoidal integration
  Vector3d gyro = (imudata.gyro + imudatalast.gyro) / 2;
  Vector3d acc = (imudata.acc + imudatalast.acc) / 2;
  imudatalast = imudata;

  Vector3d angvel_hat =
      imudata.gyro -
      error_state.bias_gyro;  // Planet angular velocity is ignored
  Matrix3d Rq_hat = nominal_state.get_imu_R();
  Vector3d acc_hat = imudata.acc - error_state.bias_acc;

  // Nominal State
  Quaterniond omg_l(0, angvel_hat.x(), angvel_hat.y(), angvel_hat.z());

  // Naive intergation
  auto qdot = nominal_state.q_imu * omg_l;
  auto vdot = Rq_hat * acc_hat + IMUData::Gravity;
  auto pdot = nominal_state.v_imu;

  // Internal the quaternion is save as [qw, qx, qy, qz] in Eigen
  nominal_state.q_imu.coeffs().block<3, 1>(0, 0) +=
      qdot.coeffs().block<3, 1>(0, 0) * dt;
  nominal_state.q_imu.normalize();
  nominal_state.v_imu += vdot * dt;
  nominal_state.p_imu += pdot * dt;

  // Error state
  // Model:
  // d (x_err)/dt = F_mat * x_err + G * n_imu
  // x_err: error state vector

  Eigen::Matrix<double, IMU_STATE_DIM, IMU_STATE_DIM> F_mat;
  F_mat.setZero();

  // Rows 1-3, dynamics on quat
  F_mat.block<3, 3>(0, 0) = skewSymmetric(angvel_hat);
  F_mat.block<3, 3>(0, 3) = -Matrix3d::Identity();

  // Rows 4-6, dynamics on bias is empty
  // Rows 7-9, dynamics on velocity
  F_mat.block<3, 3>(6, 0) = -Rq_hat * skewSymmetric(acc_hat);
  F_mat.block<3, 3>(6, 6) = -2 * skewSymmetric(acc_hat);
  F_mat.block<3, 3>(6, 9) = -Rq_hat;
  F_mat.block<3, 3>(6, 12) = -skewSymmetric(acc_hat);
  // Rows 10-12, dynamics on bias is empty

  // Rows 13-15, dynamics on position
  F_mat.block<3, 3>(12, 6) = Matrix3d::Identity();

  Eigen::Matrix<double, IMU_STATE_DIM, IMU_NOISE_DIM> G_mat;
  G_mat.setZero();
  G_mat.block<3, 3>(0, 0) = -Matrix3d::Identity();
  G_mat.block<3, 3>(3, 3) = Matrix3d::Identity();
  G_mat.block<3, 3>(6, 6) = -Rq_hat;
  G_mat.block<3, 3>(9, 9) = Matrix3d::Identity();

  // Now we need to intergate this, naive approach is trapezoidal rule
  // x_err_new = F_mat*x_err_last*dt + x_err_last
  // Or \dot Phi = F_mat Phi, Phi(0) = I
  // Phi = I + F_mat Phi * dt
  Eigen::Matrix<double, IMU_STATE_DIM, IMU_STATE_DIM> Phi;
  Phi.setIdentity();
  Phi = Phi + F_mat * dt;
  auto G = G_mat * dt;

  // Suggest by (268)-(269) in Sola J. Quaternion kinematics for the error-state
  // Kalman filter We don't predict the error state space Instead, we only
  // predict the P of error state, and predict the nominal state
  auto P_new =
      Phi * error_state.getImuP() * Phi.transpose() + G * Q_imu * G.transpose();
  auto P_imu_other_new = Phi * error_state.getImuOtherP();

  // Set states to error_state
  error_state.setImuP(P_new);
  error_state.setImuOtherP(P_imu_other_new);
}

void MSCKF::addKeyframe(const double t) {
  // For convience, we require t here is exact same to last imu t
  if (!initFirstPoseFlag) {
    if (imubuf.size() >= _config.init_imu_num) {
      initFirstPose();
    }
    return;
  }
  if (t_last >= 0) {
    assert(fabs(t - t_last) < 1.0 / _config.IMU_FREQ &&
           "MSCKF new image must be added EXACTLY after the corresponding imu "
           "is applied!");
  }
  error_state.stateAugmentation(t);
  nominal_state.addKeyframe(t);
}

void MSCKF::update(const LandmarkPerId& feature_by_id) {}

}  // namespace D2VINS