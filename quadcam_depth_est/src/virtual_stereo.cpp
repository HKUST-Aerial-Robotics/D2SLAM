#include "virtual_stereo.hpp"
#include <d2common/fisheye_undistort.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/calib3d.hpp>
#include "hitnet_onnx.hpp"
#include "crestereo_onnx.hpp"

namespace D2QuadCamDepthEst {
VirtualStereo::VirtualStereo(int _cam_idx_a, int _cam_idx_b, 
        const Swarm::Pose & baseline, 
        D2Common::FisheyeUndist* _undist_left,
        D2Common::FisheyeUndist* _undist_right,
        int _undist_id_l, 
        int _undist_id_r,
        HitnetONNX* _hitnet, CREStereoONNX* _crestereo):
    cam_idx_a(_cam_idx_a), cam_idx_b(_cam_idx_b), undist_left(_undist_left), undist_right(_undist_right),
    undist_id_l(_undist_id_l), undist_id_r(_undist_id_r), hitnet(_hitnet), crestereo(_crestereo) { 
    auto cam_param = static_cast<const camodocal::PinholeCamera*>(undist_left->cam_side.get())->getParameters();
    img_size = cv::Size(cam_param.imageWidth(), cam_param.imageHeight());
    cv::Mat K = (cv::Mat_<double>(3,3) << cam_param.fx(), 0, cam_param.cx(), 0, cam_param.fy(), cam_param.cy(), 0, 0, 1);
    initRecitfy(baseline, K, cv::Mat(), K, cv::Mat());
}

void VirtualStereo::initRecitfy(const Swarm::Pose & baseline, cv::Mat K0, cv::Mat D0, cv::Mat K1, cv::Mat D1) {
    cv::eigen2cv(baseline.R(), R);
    cv::eigen2cv(baseline.pos(), T);
    cv::stereoRectify(K0, D0, K1, D1, img_size, R, T, R1, R2, T1, T2, Q, 1024, -1, cv::Size(), &roi_l, &roi_r);
    initUndistortRectifyMap(K0, D0, R1, T1, img_size, CV_32FC1, lmap_1, lmap_2);
    initUndistortRectifyMap(K1, D1, R2, T2, img_size, CV_32FC1, rmap_1, rmap_2);
    cuda_lmap_1.upload(lmap_1);
    cuda_lmap_2.upload(lmap_2);
    cuda_rmap_1.upload(rmap_1);
    cuda_rmap_2.upload(rmap_2);

}

std::pair<cv::Mat, cv::Mat> VirtualStereo::estimatePointsViaRaw(const cv::Mat & left, const cv::Mat & right, const cv::Mat & left_color, bool show) {
    auto ret = estimateDisparityViaRaw(left, right, left_color, show);
    cv::Mat points;
    cv::reprojectImageTo3D(ret.first, points, Q, 3);
    return std::make_pair(points(roi_l), ret.second(roi_l));
}


std::pair<cv::Mat, cv::Mat>VirtualStereo::estimateDisparityViaRaw(const cv::Mat & left, const cv::Mat & right, const cv::Mat & left_color, bool show) {
    auto ret = rectifyImage(left, right);
    cv::Mat limg_rect(ret[0]), rimg_rect(ret[1]);
    auto disp = estimateDisparity(limg_rect, rimg_rect);
    if (show) {
        cv::Mat show;
        cv::Mat disp_show; //64 is max
        disp.convertTo(disp_show, CV_8U, 255.0/32.0);
        cv::applyColorMap(disp_show, disp_show, cv::COLORMAP_JET);
        cv::rectangle(disp_show, roi_l, cv::Scalar(0, 0, 255), 2);
        cv::Mat limg_rect_show, rimg_rect_show;
        limg_rect.convertTo(limg_rect_show, CV_8U);
        rimg_rect.convertTo(rimg_rect_show, CV_8U);
        cv::rectangle(limg_rect_show, roi_l, cv::Scalar(0, 0, 255), 2);
        cv::rectangle(rimg_rect_show, roi_r, cv::Scalar(0, 0, 255), 2);
        if (limg_rect.channels() == 1) {
            cv::cvtColor(limg_rect_show, limg_rect_show, cv::COLOR_GRAY2BGR);
            cv::cvtColor(rimg_rect_show, rimg_rect_show, cv::COLOR_GRAY2BGR);
        }
        cv::hconcat(limg_rect_show, rimg_rect_show, show);
        cv::hconcat(show, disp_show, show);
        char buf[64];
        sprintf(buf, "VirtualStereo %d<->%d", cam_idx_a, cam_idx_b);
        // cv::resize(show, show, cv::Size(), 2, 2);
        cv::imshow(buf, show);
    }
    if (enable_texture) {
        // cv::cuda::remap(lcolor_gpu, lcolor_gpu, cuda_lmap_1, cuda_lmap_2, cv::INTER_LINEAR);
        cv::cuda::GpuMat lcolor_gpu;
        if (left.channels() == 1) {
            lcolor_gpu = undist_left->undist_id_cuda(left_color, undist_id_l, false);
            lcolor_gpu.convertTo(lcolor_gpu, CV_8UC3);
            cv::Mat lcolor_rect(lcolor_gpu);
            return std::make_pair(disp, lcolor_rect);
        } else {
            return std::make_pair(disp, limg_rect);
        }
    } else {
        return std::make_pair(disp, limg_rect);
    }
}

std::vector<cv::cuda::GpuMat> VirtualStereo::rectifyImage(const cv::Mat & left, const cv::Mat & right) {
    cv::cuda::GpuMat leftRectify;
    cv::cuda::GpuMat rightRectify;
    auto img_cuda_l = undist_left->undist_id_cuda(left, undist_id_l, true);
    auto img_cuda_r = undist_right->undist_id_cuda(right, undist_id_r, true);
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
    if (config.use_cnn) {
        if (hitnet!=nullptr) {
            if (left.channels() == 3) {
                cv::Mat left_gray;
                cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
                return estimateDisparity(left_gray, right);
            }
            return hitnet->inference(left, right);
        } else {
            return crestereo->inference(left, right);
        }
    } else {
        return estimateDisparityOCV(left, right);
    }
}

}

