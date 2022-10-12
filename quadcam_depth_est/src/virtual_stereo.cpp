#include "virtual_stereo.hpp"
#include <d2common/fisheye_undistort.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/calib3d.hpp>
#include "hitnet_onnx.hpp"
#include "crestereo_onnx.hpp"
#include <camodocal/camera_models/CataCamera.h>
#include <opencv2/ccalib/omnidir.hpp>

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

void VirtualStereo::initVingette(const cv::Mat & _inv_vingette_l, const cv::Mat & _inv_vingette_r) {
    inv_vingette_l.upload(_inv_vingette_l);
    inv_vingette_r.upload(_inv_vingette_r);
}

VirtualStereo::VirtualStereo(const Swarm::Pose & baseline, 
            camodocal::CameraPtr cam_left,
            camodocal::CameraPtr cam_right, HitnetONNX* _hitnet, CREStereoONNX * _crestereo): 
            hitnet(_hitnet), crestereo(_crestereo) {
    cv::eigen2cv(baseline.R(), R);
    cv::eigen2cv(baseline.pos(), T);
    // int flag = cv::omnidir::RECTIFY_LONGLATI;
    int flag = cv::omnidir::RECTIFY_PERSPECTIVE;
    cv::Mat K0, K1, D0, D1;
    double xi0,xi1;
    if (cam_left->modelType() == camodocal::Camera::ModelType::MEI) {
        //Only support Mei camera
        auto cam_param = static_cast<const camodocal::CataCamera*>(cam_left.get())->getParameters();
        img_size = cv::Size(320, 240);
        double s = 0.0;
        K0 = (cv::Mat_<double>(3,3) << cam_param.gamma1(), s, cam_param.u0(), 0, cam_param.gamma2(), cam_param.v0(), 0, 0, 1);
        D0 = (cv::Mat_<double>(1,4) << cam_param.k1(), cam_param.k2(), cam_param.p1(), cam_param.p2());
        if (flag == cv::omnidir::RECTIFY_LONGLATI) {
            P1 = (cv::Mat_<double>(3,3) << img_size.width/3.1415, 0, 0,
                0, img_size.height/3.1415, 0,
                0, 0, 1);
        } else {
            P1 = (cv::Mat_<double>(3,3) << img_size.width/4, 0, img_size.width/2,
                0, img_size.height/4, img_size.height/2,
                0, 0, 1);
        }
        P2 = P1;
        xi0 = cam_param.xi();
        cam_param = static_cast<const camodocal::CataCamera*>(cam_right.get())->getParameters();
        K1 = (cv::Mat_<double>(3,3) << cam_param.gamma1(), s, cam_param.u0(), 0, cam_param.gamma2(), cam_param.v0(), 0, 0, 1);
        D1 = (cv::Mat_<double>(1,4) << cam_param.k1(), cam_param.k2(), cam_param.p1(), cam_param.p2());
        xi1 = cam_param.xi();
        cv::stereoRectify(P1, cv::Mat(), P1, cv::Mat(), img_size, R, T, R1, R2, P1, P2, Q, 1024, -1, cv::Size());
    } else {
        printf("Unsupported camera model:", cam_left->modelType());
    }
    cv::omnidir::initUndistortRectifyMap(K0, D0, xi0, R1, P1, img_size, CV_32FC1, lmap_1, lmap_2, flag);
    cv::omnidir::initUndistortRectifyMap(K1, D1, xi1, R2, P2, img_size, CV_32FC1, rmap_1, rmap_2, flag);

    cuda_lmap_1.upload(lmap_1);
    cuda_lmap_2.upload(lmap_2);
    cuda_rmap_1.upload(rmap_1);
    cuda_rmap_2.upload(rmap_2);
    input_is_stereo = true;

    // std::cout << "img_size" << img_size << std::endl;
    // std::cout << "K0: " << K0 << std::endl;
    // std::cout << "K1: " << K1 << std::endl;
    // std::cout << "D0: " << D0 << std::endl;
    // std::cout << "D1: " << D1 << std::endl;
    // std::cout << "xi0: " << xi0 << std::endl;
    // std::cout << "xi1: " << xi1 << std::endl;
    // std::cout << "R: " << R << std::endl;
    // std::cout << "T: " << T << std::endl;
    // std::cout << "R1: " << R1 << std::endl;
    // std::cout << "R2: " << R2 << std::endl;
    // std::cout << "P1: " << P1 << std::endl;
    // std::cout << "P2: " << P2 << std::endl;
    // std::cout << "Q: " << Q << std::endl;
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
    if (roi_l.empty()) {
        return std::make_pair(points, ret.second);
    }
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
        // cv::normalize(disp, disp_show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
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
        if (input_is_stereo) {
            cv::Mat lcolor;
            if (limg_rect.channels() == 1) {
                cv::cvtColor(limg_rect, lcolor, cv::COLOR_GRAY2BGR);
                return std::make_pair(disp, lcolor);
            }
            return std::make_pair(disp, limg_rect);
        }
        if (left.channels() == 1) {
            cv::cuda::GpuMat lcolor_gpu;
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
    cv::cuda::GpuMat leftRectify, rightRectify, img_cuda_l, img_cuda_r;
    if (input_is_stereo) {
        img_cuda_l.upload(left);
        img_cuda_r.upload(right);
        if (!inv_vingette_l.empty()) {
            img_cuda_l.convertTo(img_cuda_l, CV_32FC1);
            img_cuda_r.convertTo(img_cuda_r, CV_32FC1);
            cv::cuda::multiply(img_cuda_l, inv_vingette_l, img_cuda_l);
            cv::cuda::multiply(img_cuda_r, inv_vingette_r, img_cuda_r);
        }
    } else {
        auto img_cuda_l = undist_left->undist_id_cuda(left, undist_id_l, true);
        auto img_cuda_r = undist_right->undist_id_cuda(right, undist_id_r, true);
    }
    cv::cuda::remap(img_cuda_l, leftRectify, cuda_lmap_1, cuda_lmap_2, cv::INTER_LINEAR);
    cv::cuda::remap(img_cuda_r, rightRectify, cuda_rmap_1, cuda_rmap_2, cv::INTER_LINEAR);
    return {leftRectify, rightRectify};
}

cv::Mat VirtualStereo::estimateDisparityOCV(const cv::Mat & left, const cv::Mat & right) {
    auto sgbm = cv::StereoSGBM::create(config.minDisparity, config.numDisparities, config.blockSize,
        config.P1, config.P2, config.disp12MaxDiff, config.preFilterCap, config.uniquenessRatio, 
        config.speckleWindowSize, config.speckleRange, config.mode);
    cv::Mat disparity;
    if (left.type() == CV_32FC1) {
        cv::Mat left_show, right_show;
        left.convertTo(left_show, CV_8U);
        right.convertTo(right_show, CV_8U);
        sgbm->compute(left_show, right_show, disparity);
    } else {
        sgbm->compute(left, right, disparity);
    }
    disparity = disparity / 16.0;
    return disparity;
}

cv::Mat VirtualStereo::estimateDisparity(const cv::Mat & left, const cv::Mat & right) {
    if (config.use_cnn && (hitnet != nullptr || crestereo!=nullptr)) {
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

