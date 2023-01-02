#pragma once
#include <opencv2/opencv.hpp>

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <chrono>
#include <d2common/d2basetypes.h>
#include <d2frontend/d2frontend_params.h>

namespace D2FrontEnd {
using D2Common::LandmarkIdType;

cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::Image &img_msg);
cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg, int flag);
Eigen::MatrixXf load_csv_mat_eigen(std::string csv);
Eigen::VectorXf load_csv_vec_eigen(std::string csv);

template<typename T, typename B>
inline void reduceVector(std::vector<T> &v, std::vector<B> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


Swarm::Pose AffineRestoCamPose(Eigen::Matrix4d affine);

void PnPInitialFromCamPose(const Swarm::Pose &p, cv::Mat & rvec, cv::Mat & tvec);
Swarm::Pose PnPRestoCamPose(cv::Mat rvec, cv::Mat tvec);
cv::Vec3b extractColor(const cv::Mat &img, cv::Point2f p);
cv::Point2f rotate_pt_norm2d(cv::Point2f pt, Eigen::Quaterniond q);


void detectPoints(const cv::Mat & img, std::vector<cv::Point2f> & n_pts, std::vector<cv::Point2f> & cur_pts, int require_pts, bool enable_cuda=true);

std::vector<cv::cuda::GpuMat> buildImagePyramid(const cv::cuda::GpuMat& prevImg, int maxLevel_=3);

std::vector<cv::Point2f> opticalflowTrack(const cv::Mat & cur_img, const cv::Mat & prev_img, std::vector<cv::Point2f> & prev_pts, 
        std::vector<LandmarkIdType> & ids, TrackLRType type=WHOLE_IMG_MATCH, bool enable_cuda=true);

std::vector<cv::Point2f> opticalflowTrackPyr(const cv::Mat & cur_img, std::vector<cv::cuda::GpuMat> & prev_pyr, 
        std::vector<cv::Point2f> & prev_pts, std::vector<LandmarkIdType> & ids, TrackLRType type=WHOLE_IMG_MATCH, bool update_pyr=true);

std::vector<cv::DMatch> matchKNN(const cv::Mat & desc_a, const cv::Mat & desc_b, double knn_match_ratio=0.8,
        const std::vector<cv::Point2f> pts_a=std::vector<cv::Point2f>(),
        const std::vector<cv::Point2f> pts_b=std::vector<cv::Point2f>(),
        double search_local_dist = -1);
    
int computeRelativePosePnP(const std::vector<Vector3d> lm_positions_a, const std::vector<Vector3d> lm_3d_norm_b,
        Swarm::Pose extrinsic_b, Swarm::Pose drone_pose_a, Swarm::Pose drone_pose_b, Swarm::Pose & DP_b_to_a,
        std::vector<int> &inliers, bool is_4dof, bool verify_gravity=true);
Swarm::Pose computePosePnPnonCentral(const std::vector<Vector3d> & lm_positions_a, const std::vector<Vector3d> & lm_3d_norm_b,
        const std::vector<Swarm::Pose> & cam_extrinsics, const std::vector<int> & camera_indices, std::vector<int> &inliers);
int computeRelativePosePnPnonCentral(const std::vector<Vector3d> & lm_positions_a, const std::vector<Vector3d> & lm_3d_norm_b,
        const std::vector<Swarm::Pose> & cam_extrinsics, const std::vector<int> & camera_indices, 
        Swarm::Pose drone_pose_a, Swarm::Pose drone_pose_b, Swarm::Pose & DP_b_to_a,
        std::vector<int> &inliers, bool is_4dof, bool verify_gravity=true);
}