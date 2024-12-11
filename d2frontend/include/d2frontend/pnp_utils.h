#pragma once

#include <chrono>
#include <d2common/d2basetypes.h>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>

namespace D2FrontEnd {
using D2Common::LandmarkIdType;

Swarm::Pose AffineRestoCamPose(Eigen::Matrix4d affine);

void PnPInitialFromCamPose(const Swarm::Pose &p, cv::Mat &rvec, cv::Mat &tvec);
Swarm::Pose PnPRestoCamPose(cv::Mat rvec, cv::Mat tvec);
cv::Vec3b extractColor(const cv::Mat &img, cv::Point2f p);
cv::Point2f rotate_pt_norm2d(cv::Point2f pt, Eigen::Quaterniond q);

int computeRelativePosePnP(const std::vector<Vector3d> lm_positions_a,
                           const std::vector<Vector3d> lm_3d_norm_b,
                           Swarm::Pose extrinsic_b, Swarm::Pose drone_pose_a,
                           Swarm::Pose drone_pose_b, Swarm::Pose &DP_b_to_a,
                           std::vector<int> &inliers, bool is_4dof,
                           bool verify_gravity = true);
Swarm::Pose
computePosePnPnonCentral(const std::vector<Vector3d> &lm_positions_a,
                         const std::vector<Vector3d> &lm_3d_norm_b,
                         const std::vector<Swarm::Pose> &cam_extrinsics,
                         const std::vector<int> &camera_indices,
                         std::vector<int> &inliers);
int computeRelativePosePnPnonCentral(
    const std::vector<Vector3d> &lm_positions_a,
    const std::vector<Vector3d> &lm_3d_norm_b,
    const std::vector<Swarm::Pose> &cam_extrinsics,
    const std::vector<int> &camera_indices, Swarm::Pose drone_pose_a,
    Swarm::Pose drone_pose_b, Swarm::Pose &DP_b_to_a, std::vector<int> &inliers,
    bool is_4dof, bool verify_gravity = true);
} // namespace D2FrontEnd