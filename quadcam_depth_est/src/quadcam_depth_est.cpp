#include "quadcam_depth_est.hpp"
#include <d2common/fisheye_undistort.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/calib3d.hpp>


namespace D2QuadCamDepthEst {
VirtualStereo::VirtualStereo(int _idx_a, int _idx_b, 
        const Swarm::Pose & _pose_left, 
        const Swarm::Pose & _pose_right, 
        D2Common::FisheyeUndist* _undist_left,
        D2Common::FisheyeUndist* _undist_right,
        int _undist_id_l, 
        int _undist_id_r):
    idx_a(_idx_a), idx_b(_idx_b), undist_left(_undist_left), undist_right(_undist_right),
    undist_id_l(_undist_id_l), undist_id_r(_undist_id_r) { 
    Swarm::Pose vcam_rel_l = Swarm::Pose(undist_left->t[undist_id_l], Vector3d::Zero());
    Swarm::Pose vcam_rel_r = Swarm::Pose(undist_right->t[undist_id_r], Vector3d::Zero());
    pose_left = _pose_left * vcam_rel_l;
    pose_right = _pose_right * vcam_rel_r;

    //Initialize rectified poses.
    Swarm::Pose p_l_to_r = pose_left.inverse() * pose_right;
    printf("[VirtualStereo] idx: %d<->%d, P_l_to_r: %s pose_left: %s, pose_right: %s \n", idx_a, idx_b, p_l_to_r.toStr().c_str(), 
            pose_left.toStr().c_str(), pose_right.toStr().c_str());
    //Undistortors have same pinhole2 camera parameters
    auto cam_param = static_cast<const camodocal::PinholeCamera*>(undist_left->cam_side.get())->getParameters();
    img_size = cv::Size(cam_param.imageWidth(), cam_param.imageHeight());
    cv::Mat K = (cv::Mat_<double>(3,3) << cam_param.fx(), 0, cam_param.cx(), 0, cam_param.fy(), cam_param.cy(), 0, 0, 1);
    cv::eigen2cv(p_l_to_r.R(), R);
    cv::eigen2cv(p_l_to_r.pos(), T);
    cv::stereoRectify(K, cv::Mat(), K, cv::Mat(), img_size, R, T, R1, R2, T1, T2, Q);
    std::cout << "R1" << std::endl << R1 << std::endl;
    std::cout << "R2" << std::endl << R2 << std::endl;
    //Initial maps
    initUndistortRectifyMap(K, cv::Mat(), R1, T1, img_size, CV_32FC1, lmap_1, lmap_2);
    initUndistortRectifyMap(K, cv::Mat(), R2, T2, img_size, CV_32FC1, rmap_1, rmap_2);
    cuda_lmap_1.upload(lmap_1);
    cuda_lmap_2.upload(lmap_2);
    cuda_rmap_1.upload(rmap_1);
    cuda_rmap_2.upload(rmap_2);
}

std::vector<cv::cuda::GpuMat> VirtualStereo::rectifyImage(const cv::Mat & left, const cv::Mat & right) {
    cv::cuda::GpuMat leftRectify;
    cv::cuda::GpuMat rightRectify;
    auto img_cuda_l = undist_left->undist_id_cuda(left, undist_id_l);
    auto img_cuda_r = undist_right->undist_id_cuda(right, undist_id_r);
    // if (img_cuda_l.channels() == 3) {
    //     cv::cuda::cvtColor(img_cuda_l, img_cuda_l, cv::COLOR_BGR2GRAY);
    // }
    // if (img_cuda_r.channels() == 3) {
    //     cv::cuda::cvtColor(img_cuda_r, img_cuda_r, cv::COLOR_BGR2GRAY);
    // }
    cv::cuda::remap(img_cuda_l, leftRectify, cuda_lmap_1, cuda_lmap_2, cv::INTER_LINEAR);
    cv::cuda::remap(img_cuda_r, rightRectify, cuda_rmap_1, cuda_rmap_2, cv::INTER_LINEAR);
    return {leftRectify, rightRectify};
}

cv::Mat VirtualStereo::estimateDisparityOCV(const cv::Mat & left, const cv::Mat & right) {
    auto sgbm = cv::StereoSGBM::create(config.minDisparity, config.numDisparities, config.blockSize,
        config.P1, config.P2, config.disp12MaxDiff, config.preFilterCap, config.uniquenessRatio, 
        config.speckleWindowSize, config.speckleRange, config.mode);
    auto ret = rectifyImage(left, right);
    cv::Mat disparity;
    sgbm->compute(cv::Mat(ret[0]), cv::Mat(ret[1]), disparity);
    return disparity;
}

}

