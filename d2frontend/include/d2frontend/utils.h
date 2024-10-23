#pragma once
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <d2common/d2basetypes.h>
#include <d2frontend/d2frontend_params.h>
#include <d2common/d2landmarks.h>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>

namespace D2FrontEnd {
using D2Common::LandmarkIdType;
using LandmarkType = D2Common::LandmarkType;

template <typename T> struct LKImageInfo {
    std::vector<cv::Point2f> lk_pts;
    std::vector<Eigen::Vector3d> lk_pts_3d_norm;
    std::vector<LandmarkIdType> lk_ids;
    std::vector<int> lk_local_index;
    std::vector<LandmarkType> lk_types;
    std::vector<T> pyr;
};

using LKImageInfoCPU = LKImageInfo<cv::Mat>;
using LKImageInfoGPU = LKImageInfo<cv::cuda::GpuMat>;

cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::Image &img_msg);
cv_bridge::CvImagePtr
getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg,
                        int flag);
Eigen::MatrixXf load_csv_mat_eigen(std::string csv);
Eigen::VectorXf load_csv_vec_eigen(std::string csv);

template <typename T, typename B>
inline void reduceVector(std::vector<T> &v, std::vector<B> status) {
    int j = 0;
    for (unsigned int i = 0; i < v.size(); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

Swarm::Pose AffineRestoCamPose(Eigen::Matrix4d affine);

void PnPInitialFromCamPose(const Swarm::Pose &p, cv::Mat &rvec, cv::Mat &tvec);
Swarm::Pose PnPRestoCamPose(cv::Mat rvec, cv::Mat tvec);
cv::Vec3b extractColor(const cv::Mat &img, cv::Point2f p);
cv::Point2f rotate_pt_norm2d(cv::Point2f pt, Eigen::Quaterniond q);

void detectPoints(const cv::Mat &img, std::vector<cv::Point2f> &n_pts,
                  const std::vector<cv::Point2f> &cur_pts, int require_pts,
                  bool enable_cuda = true, bool use_fast = false,
                  int fast_rows = 3, int fast_cols = 4);

std::vector<cv::cuda::GpuMat> buildImagePyramid(const cv::cuda::GpuMat &prevImg,
                                                int maxLevel_ = 3);

std::vector<cv::Point2f> opticalflowTrack(const cv::Mat &cur_img,
                                          const cv::Mat &prev_img,
                                          std::vector<cv::Point2f> &prev_pts,
                                          std::vector<LandmarkIdType> &ids,
                                          TrackLRType type = WHOLE_IMG_MATCH,
                                          bool enable_cuda = true);

LKImageInfoGPU opticalflowTrackPyr(
    const cv::Mat &cur_img, const LKImageInfoGPU& prev_lk,
    TrackLRType type = WHOLE_IMG_MATCH);

std::vector<cv::DMatch>
matchKNN(const cv::Mat &desc_a, const cv::Mat &desc_b,
         double knn_match_ratio = 0.8,
         const std::vector<cv::Point2f> pts_a = std::vector<cv::Point2f>(),
         const std::vector<cv::Point2f> pts_b = std::vector<cv::Point2f>(),
         double search_local_dist = -1);

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