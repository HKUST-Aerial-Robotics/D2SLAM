#include "virtual_stereo.hpp"
#include <d2common/fisheye_undistort.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/calib3d.hpp>
#include "hitnet_onnx.hpp"

namespace D2QuadCamDepthEst {
VirtualStereo::VirtualStereo(int _cam_idx_a, int _cam_idx_b, 
        const Swarm::Pose & baseline, 
        D2Common::FisheyeUndist* _undist_left,
        D2Common::FisheyeUndist* _undist_right,
        int _undist_id_l, 
        int _undist_id_r,
        HitnetONNX* _hitnet):
    cam_idx_a(_cam_idx_a), cam_idx_b(_cam_idx_b), undist_left(_undist_left), undist_right(_undist_right),
    undist_id_l(_undist_id_l), undist_id_r(_undist_id_r), hitnet(_hitnet) { 

    //Initialize rectified poses.
    // printf("[VirtualStereo] idx: %d<->%d, P_l_to_r: %s pose_left: %s, pose_right: %s \n", cam_idx_a, cam_idx_b, p_l_to_r.toStr().c_str(), 
    //         pose_left.toStr().c_str(), pose_right.toStr().c_str());
    //Undistortors have same pinhole2 camera parameters
    auto cam_param = static_cast<const camodocal::PinholeCamera*>(undist_left->cam_side.get())->getParameters();
    img_size = cv::Size(cam_param.imageWidth(), cam_param.imageHeight());
    cv::Mat K = (cv::Mat_<double>(3,3) << cam_param.fx(), 0, cam_param.cx(), 0, cam_param.fy(), cam_param.cy(), 0, 0, 1);
    cv::eigen2cv(baseline.R(), R);
    cv::eigen2cv(baseline.pos(), T);
    cv::stereoRectify(K, cv::Mat(), K, cv::Mat(), img_size, R, T, R1, R2, T1, T2, Q);
    // std::cout << "R1" << std::endl << R1 << std::endl;
    // std::cout << "R2" << std::endl << R2 << std::endl;
    //Initial maps
    initUndistortRectifyMap(K, cv::Mat(), R1, T1, img_size, CV_32FC1, lmap_1, lmap_2);
    initUndistortRectifyMap(K, cv::Mat(), R2, T2, img_size, CV_32FC1, rmap_1, rmap_2);
    cuda_lmap_1.upload(lmap_1);
    cuda_lmap_2.upload(lmap_2);
    cuda_rmap_1.upload(rmap_1);
    cuda_rmap_2.upload(rmap_2);
}

std::pair<cv::Mat, cv::Mat> VirtualStereo::estimatePointsViaRaw(const cv::Mat & left, const cv::Mat & right, bool show) {
    auto ret = estimateDisparityViaRaw(left, right, show);
    cv::Mat points;
    cv::reprojectImageTo3D(ret.first, points, Q, 3);
    return std::make_pair(points, ret.second);
}


std::pair<cv::Mat, cv::Mat>VirtualStereo::estimateDisparityViaRaw(const cv::Mat & left, const cv::Mat & right, bool show) {
    auto limg = undist_left->undist_id_cuda(left, undist_id_l);
    auto rimg = undist_right->undist_id_cuda(right, undist_id_r);
    if (!enable_texture) {
        cv::cuda::cvtColor(limg, limg, cv::COLOR_BGR2GRAY);
    }
    cv::cuda::cvtColor(rimg, rimg, cv::COLOR_BGR2GRAY);
    cv::cuda::remap(limg, limg, cuda_lmap_1, cuda_lmap_2, cv::INTER_LINEAR);
    cv::cuda::remap(rimg, rimg, cuda_rmap_1, cuda_rmap_2, cv::INTER_LINEAR);
    cv::Mat limg_rect(limg), rimg_rect(rimg);
    auto disp = estimateDisparity(limg_rect, rimg_rect);
    if (show) {
        cv::Mat show;
        cv::Mat disp_show; //64 is max
        disp.convertTo(disp_show, CV_8U, 255.0/32.0);
        cv::applyColorMap(disp_show, disp_show, cv::COLORMAP_JET);
        if (!enable_texture) {
            cv::cvtColor(limg_rect, limg_rect, cv::COLOR_GRAY2BGR);
        }
        cv::hconcat(limg_rect, disp_show, show);
        char buf[64];
        sprintf(buf, "VirtualStereo %d<->%d", cam_idx_a, cam_idx_b);
        cv::imshow(buf, show);
    }
    if (enable_texture) {
        return std::make_pair(disp, limg_rect);
    } else {
        return std::make_pair(disp, cv::Mat());
    }
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
    disparity = disparity / 16.0;
    return disparity;
}

cv::Mat VirtualStereo::estimateDisparity(const cv::Mat & left, const cv::Mat & right) {
    if (left.channels() == 3) {
        cv::Mat left_gray;
        cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
        return estimateDisparity(left_gray, right);
    }
    if (config.use_hitnet) {
        return hitnet->inference(left, right);
    } else {
        return estimateDisparityOCV(left, right);
    }
}

}

