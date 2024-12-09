#include <d2common/d2basetypes.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/loop_detector.h>
#include <d2frontend/pnp_utils.h>
#include <spdlog/spdlog.h>

#include <d2common/utils.hpp>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

using namespace std::chrono;
using namespace D2Common;
using D2Common::Utility::TicToc;

#define WIN_SIZE cv::Size(21, 21)

namespace D2FrontEnd {

Swarm::Pose AffineRestoCamPose(Eigen::Matrix4d affine) {
  Eigen::Matrix3d R;
  Eigen::Vector3d T;

  R = affine.block<3, 3>(0, 0);
  T = affine.block<3, 1>(0, 3);

  R = (R.normalized()).transpose();
  T = R * (-T);

  std::cout << "R of affine\n" << R << std::endl;
  std::cout << "T of affine\n" << T << std::endl;
  std::cout << "RtR\n" << R.transpose() * R << std::endl;
  return Swarm::Pose(R, T);
}

void PnPInitialFromCamPose(const Swarm::Pose &p, cv::Mat &rvec, cv::Mat &tvec) {
  Eigen::Matrix3d R_w_c = p.R();
  Eigen::Matrix3d R_inital = R_w_c.inverse();
  Eigen::Vector3d T_w_c = p.pos();
  cv::Mat tmp_r;
  Eigen::Vector3d P_inital = -(R_inital * T_w_c);

  cv::eigen2cv(R_inital, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_inital, tvec);
}

Swarm::Pose PnPRestoCamPose(cv::Mat rvec, cv::Mat tvec) {
  cv::Mat r;
  cv::Rodrigues(rvec, r);
  Eigen::Matrix3d R_pnp, R_w_c_old;
  cv::cv2eigen(r, R_pnp);
  R_w_c_old = R_pnp.transpose();
  Eigen::Vector3d T_pnp, T_w_c_old;
  cv::cv2eigen(tvec, T_pnp);
  T_w_c_old = R_w_c_old * (-T_pnp);

  return Swarm::Pose(R_w_c_old, T_w_c_old);
}

bool pnp_result_verify(bool pnp_success, int inliers, double rperr,
                       const Swarm::Pose &DP_old_to_new) {
  bool success = pnp_success;
  if (!pnp_success) {
    return false;
  }
  if (rperr > params->loopdetectorconfig->gravity_check_thres) {
    printf("[SWARM_LOOP] Check failed on RP error %.1fdeg (%.1f)deg\n",
           rperr * 57.3,
           params->loopdetectorconfig->gravity_check_thres * 57.3);
    return false;
  }
  auto &_config = (*params->loopdetectorconfig);
  success = (inliers >= _config.loop_inlier_feature_num) &&
            fabs(DP_old_to_new.yaw()) < _config.accept_loop_max_yaw * DEG2RAD &&
            DP_old_to_new.pos().norm() < _config.accept_loop_max_pos;
  return success;
}

double gravityCheck(const Swarm::Pose &pnp_pose,
                    const Swarm::Pose &ego_motion) {
  // This checks the gravity direction
  Vector3d gravity(0, 0, 1);
  Vector3d gravity_pnp = pnp_pose.R().inverse() * gravity;
  Vector3d gravity_ego = ego_motion.R().inverse() * gravity;
  double sin_theta = gravity_pnp.cross(gravity_ego).norm();
  return sin_theta;
}

int computeRelativePosePnP(const std::vector<Vector3d> lm_positions_a,
                           const std::vector<Vector3d> lm_3d_norm_b,
                           Swarm::Pose extrinsic_b, Swarm::Pose ego_motion_a,
                           Swarm::Pose ego_motion_b, Swarm::Pose &DP_b_to_a,
                           std::vector<int> &inliers, bool is_4dof,
                           bool verify_gravity) {
  // Compute PNP
  // ROS_INFO("Matched features %ld", matched_2d_norm_old.size());
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
  cv::Mat r, rvec, rvec2, t, t2, D, tmp_r;

  int iteratives = 100;
  Point3fVector pts3d;
  Point2fVector pts2d;
  for (unsigned int i = 0; i < lm_positions_a.size(); i++) {
    auto z = lm_3d_norm_b[i].z();
    if (z > 1e-1) {
      pts3d.push_back(cv::Point3f(lm_positions_a[i].x(), lm_positions_a[i].y(),
                                  lm_positions_a[i].z()));
      pts2d.push_back(
          cv::Point2f(lm_3d_norm_b[i].x() / z, lm_3d_norm_b[i].y() / z));
    }
  }
  if (pts3d.size() < params->loopdetectorconfig->loop_inlier_feature_num) {
    return false;
  }
  bool success = solvePnPRansac(pts3d, pts2d, K, D, rvec, t, false, iteratives,
                                5.0 / params->focal_length, 0.99, inliers);
  auto p_cam_old_in_new = PnPRestoCamPose(rvec, t);
  auto pnp_predict_pose_b =
      p_cam_old_in_new * (extrinsic_b.toIsometry().inverse());
  if (!success) {
    return 0;
  }
  DP_b_to_a = Swarm::Pose::DeltaPose(pnp_predict_pose_b, ego_motion_a, is_4dof);
  if (verify_gravity) {
    auto RPerr = gravityCheck(pnp_predict_pose_b, ego_motion_b);
    success = pnp_result_verify(success, inliers.size(), RPerr, DP_b_to_a);
    printf(
        "[SWARM_LOOP@%d] DPose %s PnPRansac %d inlines %d/%d, dyaw %f "
        "dpos %f g_err %f \n",
        params->self_id, DP_b_to_a.toStr().c_str(), success, inliers.size(),
        pts2d.size(), fabs(DP_b_to_a.yaw()) * 57.3, DP_b_to_a.pos().norm(),
        RPerr);
  }
  return success;
}

Swarm::Pose computePosePnPnonCentral(
    const std::vector<Vector3d> &lm_positions_a,
    const std::vector<Vector3d> &lm_3d_norm_b,
    const std::vector<Swarm::Pose> &cam_extrinsics,
    const std::vector<int> &camera_indices, std::vector<int> &inliers) {
  opengv::bearingVectors_t bearings;
  std::vector<int> camCorrespondences;
  opengv::points_t points;
  opengv::rotations_t camRotations;
  opengv::translations_t camOffsets;
  for (unsigned int i = 0; i < lm_positions_a.size(); i++) {
    bearings.push_back(lm_3d_norm_b[i]);
    camCorrespondences.push_back(camera_indices[i]);
    points.push_back(lm_positions_a[i]);
  }
  for (unsigned int i = 0; i < cam_extrinsics.size(); i++) {
    camRotations.push_back(cam_extrinsics[i].R());
    camOffsets.push_back(cam_extrinsics[i].pos());
  }

  // Solve with GP3P + RANSAC
  opengv::absolute_pose::NoncentralAbsoluteAdapter adapter(
      bearings, camCorrespondences, points, camOffsets, camRotations);
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  // ransac.threshold_ = 1.0 - cos(atan(sqrt(10.0)*0.5/460.0));
  ransac.threshold_ = 0.5 / params->focal_length;
  ransac.max_iterations_ = 50;
  ransac.computeModel();
  // Obtain relative pose results
  inliers = ransac.inliers_;
  auto best_transformation = ransac.model_coefficients_;
  Matrix3d R = best_transformation.block<3, 3>(0, 0);
  Vector3d t = best_transformation.block<3, 1>(0, 3);
  Swarm::Pose p_drone_old_in_new_init(R, t);

  // Filter by inliers and perform non-linear optimization to refine.
  std::set<int> inlier_set(inliers.begin(), inliers.end());
  bearings.clear();
  camCorrespondences.clear();
  points.clear();
  for (unsigned int i = 0; i < lm_positions_a.size(); i++) {
    if (inlier_set.find(i) == inlier_set.end()) {
      continue;
    }
    bearings.push_back(lm_3d_norm_b[i]);
    camCorrespondences.push_back(camera_indices[i]);
    points.push_back(lm_positions_a[i]);
  }
  adapter.sett(t);
  adapter.setR(R);
  opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter);
  R = nonlinear_transformation.block<3, 3>(0, 0);
  t = nonlinear_transformation.block<3, 1>(0, 3);
  Swarm::Pose p_drone_old_in_new(R, t);
  // printf("[InitPnP] pose_init %s pose_refine %s\n",
  // p_drone_old_in_new_init.toStr().c_str(),
  //         p_drone_old_in_new.toStr().c_str());
  return p_drone_old_in_new;
}

int computeRelativePosePnPnonCentral(
    const std::vector<Vector3d> &lm_positions_a,
    const std::vector<Vector3d> &lm_3d_norm_b,
    const std::vector<Swarm::Pose> &cam_extrinsics,
    const std::vector<int> &camera_indices, Swarm::Pose drone_pose_a,
    Swarm::Pose ego_motion_b, Swarm::Pose &DP_b_to_a, std::vector<int> &inliers,
    bool is_4dof, bool verify_gravity) {
  D2Common::Utility::TicToc tic;
  auto pnp_predict_pose_b = computePosePnPnonCentral(
      lm_positions_a, lm_3d_norm_b, cam_extrinsics, camera_indices, inliers);
  DP_b_to_a = Swarm::Pose::DeltaPose(pnp_predict_pose_b, drone_pose_a, is_4dof);

  bool success = true;
  double RPerr = 0;
  if (verify_gravity) {
    // Verify the results
    auto RPerr = gravityCheck(pnp_predict_pose_b, ego_motion_b);
    success = pnp_result_verify(true, inliers.size(), RPerr, DP_b_to_a);
  }

  SPDLOG_INFO(
      "[LoopDetector@{}] features {}/{} succ {} gPnPRansac time {:.2f}ms "
      "RP: {} g_err P{}\n",
      params->self_id, inliers.size(), lm_3d_norm_b.size(), success, tic.toc(),
      DP_b_to_a.toStr(), RPerr);
  return success;
}

}  // namespace D2FrontEnd