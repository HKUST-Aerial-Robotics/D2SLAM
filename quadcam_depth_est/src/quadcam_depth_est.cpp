#include "quadcam_depth_est.hpp"
#include <d2common/fisheye_undistort.h>
#include "hitnet_onnx.hpp"
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>


namespace D2FrontEnd {
    std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config);
};

namespace D2QuadCamDepthEst {
std::pair<cv::Mat, cv::Mat> intrinsicsFromNode(const YAML::Node & node);

QuadCamDepthEst::QuadCamDepthEst(ros::NodeHandle & _nh): nh(_nh) {
    std::string quadcam_depth_config_file;
    nh.param<std::string>("quadcam_depth_config_file", quadcam_depth_config_file, "quadcam_depth_config.yaml");
    nh.param<bool>("show", show, false);
    pub_pcl = nh.advertise<sensor_msgs::PointCloud2>("/depth_estimation/point_cloud", 1);
    printf("[QuadCamDepthEst] readConfig from: %s\n", quadcam_depth_config_file.c_str());
    int pn = quadcam_depth_config_file.find_last_of('/');    
    std::string configPath = quadcam_depth_config_file.substr(0, pn);
    YAML::Node config = YAML::LoadFile(quadcam_depth_config_file);
    enable_texture = config["enable_texture"].as<bool>();
    pixel_step = config["pixel_step"].as<int>();
    image_step = config["image_step"].as<int>();
    min_z = config["min_z"].as<double>();
    max_z = config["max_z"].as<double>();
    loadCNN(config);
    loadCameraConfig(config, configPath);
    std::string format = "compressed";
    image_transport::TransportHints hints(format, ros::TransportHints().tcpNoDelay(true));
    it_ = new image_transport::ImageTransport(nh);
    image_sub = it_->subscribe("/arducam/image", 1000, &QuadCamDepthEst::imageCallback, this, hints);
    if (enable_texture) {
        pcl_color = new PointCloudRGB;
        pcl_color->points.reserve(virtual_stereos.size()*width*height);
    } else {
        pcl = new PointCloud;
        pcl->points.reserve(virtual_stereos.size()*width*height);
    }
}

void QuadCamDepthEst::loadCNN(YAML::Node & config) {
    bool enable_hitnet;
    std::string quadcam_depth_config_file;
    enable_hitnet = config["enable_hitnet"].as<bool>();
    width = config["width"].as<int>();
    height = config["height"].as<int>();
    if (enable_hitnet) {
        std::string hitnet_model_path;
        bool hitnet_use_tensorrt = config["hitnet_use_tensorrt"].as<bool>();
        bool hitnet_int8 = config["hitnet_int8"].as<bool>();
        bool hitnet_fp16 = config["hitnet_fp16"].as<bool>();
        nh.param<std::string>("hitnet_model_path", hitnet_model_path, "");
        hitnet = new HitnetONNX(hitnet_model_path, width, height,
            hitnet_use_tensorrt, hitnet_fp16, hitnet_int8);
        printf("[QuadCamDepthEst] Load hitnet from %s tensorrt %d FP16 %d INT8 %d width %d height %d\n", hitnet_model_path.c_str(), 
                hitnet_use_tensorrt, hitnet_fp16, hitnet_int8, width, height);
    } else {
        printf("[QuadCamDepthEst] Hitnet disabled. Use OpenCV SGBM\n");
    }
}

void QuadCamDepthEst::imageCallback(const sensor_msgs::ImageConstPtr & left) {
    if (image_count % image_step != 0) {
        image_count++;
        return;
    }
    TicToc t;
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(left, sensor_msgs::image_encodings::BGR8);
    cv::Mat img = cv_ptr->image;
    std::vector<cv::Mat> imgs;
    std::vector<cv::Mat> imgs_gray;
    const int num_imgs = 4;
    for (int i = 0; i < 4; i++) {
        imgs.emplace_back(img(cv::Rect(i * img.cols /num_imgs, 0, img.cols /num_imgs, img.rows)));
        cv::Mat img_gray;
        cv::cvtColor(imgs[i], img_gray, cv::COLOR_BGR2GRAY);
        imgs_gray.emplace_back(img_gray);
    }
    if (enable_texture) {
        pcl_conversions::toPCL(left->header.stamp, pcl_color->header.stamp);
        pcl_color->header.frame_id = "imu";
        pcl_color->points.clear();
    } else {
        pcl_conversions::toPCL(left->header.stamp, pcl->header.stamp);
        pcl->header.frame_id = "imu";
        pcl->points.clear();
    }
    int size = 0;
    for (auto stereo: virtual_stereos) {
        auto ret = stereo->estimatePointsViaRaw(imgs_gray[stereo->cam_idx_a], imgs_gray[stereo->cam_idx_b], imgs[stereo->cam_idx_a], show);
        if (enable_texture) {
            addPointsToPCL(ret.first, ret.second, stereo->extrinsic, *pcl_color, pixel_step, min_z, max_z);
        } else {
            addPointsToPCL(ret.first, ret.second, stereo->extrinsic, *pcl, pixel_step, min_z, max_z);
        }
    }
    if (show) {
        cv::waitKey(1);
    }
    if (enable_texture) {
        pub_pcl.publish(*pcl_color);
    } else {
        pub_pcl.publish(*pcl);
    }
    image_count++;
    printf("[QuadCamDepthEst] count %d process time %.1fms\n", image_count, t.toc());
}


void QuadCamDepthEst::loadCameraConfig(YAML::Node & config, std::string configPath) {
    std::string mask_file = configPath + "/" + config["lightness_mask"].as<std::string>();
    cv::Mat lightness_mask = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);
    lightness_mask.convertTo(lightness_mask, CV_32FC1, 1.0/255.0);
    cv::Mat photometric;
    //inverse lightness mask to get photometric mask
    cv::divide(1.0, lightness_mask, photometric);

    std::string calib_file_path = config["calib_file_path"].as<std::string>();
    double fov = config["fov"].as<double>();
    printf("[QuadCamDepthEst] Load camera config from %s\n", calib_file_path.c_str());
    calib_file_path = configPath + "/" + calib_file_path;
    YAML::Node config_cams = YAML::LoadFile(calib_file_path);
    for (const auto& kv : config_cams) {
        std::string camera_name = kv.first.as<std::string>();
        printf("[QuadCamDepthEst] Load camera %s\n", camera_name.c_str());
        const YAML::Node& value = kv.second;
        auto ret = D2FrontEnd::readCameraConfig(camera_name, value);
        undistortors.push_back(new D2Common::FisheyeUndist(ret.first, 0, fov, true,
            D2Common::FisheyeUndist::UndistortPinhole2, width, height, photometric));
        raw_cam_extrinsics.emplace_back(ret.second);
    }


    for (const auto & kv: config["stereos"]) {
        auto node = kv.second;
        std::string stereo_name = kv.first.as<std::string>();
        int cam_idx_l = node["cam_idx_l"].as<int>();
        int cam_idx_r = node["cam_idx_r"].as<int>();
        int idx_l = node["idx_l"].as<int>();
        int idx_r = node["idx_r"].as<int>();
        std::string stereo_calib_file = configPath + "/" + node["stereo_config"].as<std::string>();
        Swarm::Pose baseline;
        YAML::Node stereo_calib = YAML::LoadFile(stereo_calib_file);
        Matrix4d T;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                T(i, j) = stereo_calib["cam1"]["T_cn_cnm1"][i][j].as<double>();
            }
        }
        baseline = Swarm::Pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
        auto KD0 = intrinsicsFromNode(stereo_calib["cam0"]);
        auto KD1 = intrinsicsFromNode(stereo_calib["cam1"]);

        printf("[QuadCamDepthEst] Load stereo %s, stereo %d(%d):%d(%d) baseline: %s\n", 
            stereo_name.c_str(), cam_idx_l, idx_l, cam_idx_r, idx_r, baseline.toStr().c_str());
        auto stereo = new VirtualStereo(cam_idx_l, cam_idx_r, baseline, 
            undistortors[cam_idx_l], undistortors[cam_idx_r], idx_l, idx_r, hitnet);
        auto att = undistortors[cam_idx_l]->t[idx_l];
        stereo->extrinsic = raw_cam_extrinsics[cam_idx_l] * Swarm::Pose(att, Vector3d(0, 0, 0));
        stereo->enable_texture = enable_texture;
        stereo->initRecitfy(baseline, KD0.first, KD0.second, KD1.first, KD1.second);
        virtual_stereos.emplace_back(stereo);
    }
}

std::pair<cv::Mat, cv::Mat> intrinsicsFromNode(const YAML::Node & node) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
    K.at<double>(0, 0) = node["intrinsics"][0].as<double>();
    K.at<double>(1, 1) = node["intrinsics"][1].as<double>();
    K.at<double>(0, 2) = node["intrinsics"][2].as<double>();
    K.at<double>(1, 2) = node["intrinsics"][3].as<double>();

    cv::Mat D = cv::Mat::zeros(4, 1, CV_64FC1);
    D.at<double>(0, 0) = node["distortion_coeffs"][0].as<double>();
    D.at<double>(1, 0) = node["distortion_coeffs"][1].as<double>();
    D.at<double>(2, 0) = node["distortion_coeffs"][2].as<double>();
    D.at<double>(3, 0) = node["distortion_coeffs"][3].as<double>();
    return std::make_pair(K, D);
}
 
}

