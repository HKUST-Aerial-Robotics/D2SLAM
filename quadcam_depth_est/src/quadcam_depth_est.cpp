#include "quadcam_depth_est.hpp"
#include <d2common/fisheye_undistort.h>
#include "hitnet_onnx.hpp"
#include "crestereo_onnx.hpp"
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

std::string arducam_topic = "/arducam/image";
std::string oak_ffc_4p_topic = "/oak_ffc_4p/assemble_image";


namespace D2FrontEnd {
    std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config);
};

namespace D2QuadCamDepthEst {
std::pair<cv::Mat, cv::Mat> intrinsicsFromNode(const YAML::Node & node);

QuadCamDepthEst::QuadCamDepthEst(ros::NodeHandle & _nh): nh(_nh) {
    std::string quadcam_depth_config_file;
    nh.getParam("depth_config", quadcam_depth_config_file);
    nh.getParam("show", show);
    // nh.param<bool>("show", show, false);
    printf("[QuadCamDepthEst] show %d\n", show);
    pub_pcl = nh.advertise<sensor_msgs::PointCloud2>("/depth_estimation/point_cloud", 1);
    printf("[QuadCamDepthEst] readConfig from: %s\n", quadcam_depth_config_file.c_str());
    int pn = quadcam_depth_config_file.find_last_of('/');    
    std::string configPath = quadcam_depth_config_file.substr(0, pn);
    YAML::Node config = YAML::LoadFile(quadcam_depth_config_file);
    this->enable_texture = config["enable_texture"].as<bool>();
    this->pixel_step = config["pixel_step"].as<int>();
    this->image_step = config["image_step"].as<int>();
    this->min_z = config["min_z"].as<double>();
    this->max_z = config["max_z"].as<double>();
    this->width = config["width"].as<int>();
    this->height = config["height"].as<int>();
    printf("caemra config: %s\n",configPath.c_str());
    loadCNN(config);
    loadCameraConfig(config, configPath);

    std::string format = "raw"; //TODO: make it configurable
    image_transport::TransportHints hints(format, ros::TransportHints().tcpNoDelay(true));
    it_ = new image_transport::ImageTransport(nh);
    if (camera_config == CameraConfig::FOURCORNER_FISHEYE) {
        image_sub = it_->subscribe(oak_ffc_4p_topic, 1000, &QuadCamDepthEst::imageCallback, this, hints);
        printf("[DEBUG]subscribe to %s\n",oak_ffc_4p_topic.c_str());
    } else {
        left_sub = new ImageSubscriber(*it_, "/cam0/image_raw", 1000, hints);
        right_sub = new ImageSubscriber(*it_, "/cam1/image_raw", 1000, hints);
        sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> (*left_sub, *right_sub, 1000);
        sync->registerCallback(boost::bind(&QuadCamDepthEst::stereoImagesCallback, this, _1, _2));
    }
    if (enable_texture) {
        pcl_color = new PointCloudRGB;
        pcl_color->points.reserve(virtual_stereos.size()*width*height);
    } else {
        pcl = new PointCloud;
        pcl->points.reserve(virtual_stereos.size()*width*height);
    }
}

void QuadCamDepthEst::loadCNN(YAML::Node & config) {
    std::string quadcam_depth_config_file;
    enable_cnn = config["enable_cnn"].as<bool>();
    std::string cnn_type = config["cnn_type"].as<std::string>();
    width = config["width"].as<int>();
    height = config["height"].as<int>();
    if (enable_cnn) {
        bool cnn_use_tensorrt = config["cnn_use_tensorrt"].as<bool>();
        bool cnn_int8 = config["cnn_int8"].as<bool>();
        bool cnn_fp16 = config["cnn_fp16"].as<bool>();
        std::string cnn_model_path = config["cnn_model_path"].as<std::string>();
        std::string quantization_file = config["cnn_quant_path"].as<std::string>();
        if (cnn_type == "hitnet") {
            hitnet = new HitnetONNX(cnn_model_path, quantization_file,width, height, cnn_use_tensorrt, cnn_fp16, cnn_int8);
            printf("[DEBUG] Hit NET\n");            
            cnn_rgb = false;
        } 
        if (cnn_type == "crestereo") {
            crestereo = new CREStereoONNX(cnn_model_path, width, height, cnn_use_tensorrt, cnn_fp16, cnn_int8);
            cnn_rgb = true;
        }
        printf("[QuadCamDepthEst] Load CNN from %s tensorrt %d FP16 %d INT8 %d width %d height %d\n", cnn_model_path.c_str(), 
                cnn_use_tensorrt, cnn_fp16, cnn_int8, width, height);

    }else {
        printf("[QuadCamDepthEst] CNN disabled. Use OpenCV SGBM\n");
    }
}

void QuadCamDepthEst::stereoImagesCallback(const sensor_msgs::ImageConstPtr left, const sensor_msgs::ImageConstPtr right) {
     if (image_count % image_step != 0) {
        image_count++;
        return;
    }
    TicToc t;
    cv_bridge::CvImagePtr cv_ptr_l = cv_bridge::toCvCopy(left, sensor_msgs::image_encodings::MONO8);
    cv_bridge::CvImagePtr cv_ptr_r = cv_bridge::toCvCopy(right, sensor_msgs::image_encodings::MONO8);
    if (enable_texture) {
        pcl_conversions::toPCL(left->header.stamp, pcl_color->header.stamp);
        pcl_color->header.frame_id = "imu";
        pcl_color->points.clear();
    } else {
        pcl_conversions::toPCL(left->header.stamp, pcl->header.stamp);
        pcl->header.frame_id = "imu";
        pcl->points.clear();
    }
    std::pair<cv::Mat, cv::Mat> ret = virtual_stereos[0]->estimatePointsViaRaw(cv_ptr_l->image, cv_ptr_r->image, cv_ptr_l->image, show);
    if (enable_texture) {
        printf("[Debug] texture\n");
        addPointsToPCL(ret.first, ret.second, virtual_stereos[0]->extrinsic, *pcl_color, pixel_step, min_z, max_z);
    } else {
        addPointsToPCL(ret.first, ret.second, virtual_stereos[0]->extrinsic, *pcl, pixel_step, min_z, max_z);
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
    cv::imshow("receive",img);
    printf("[Debug] image size %d %d\n", img.cols, img.rows);
    for (int i = 0; i < 4; i++) {
        imgs.emplace_back(img(cv::Rect(i * img.cols /num_imgs, 0, img.cols /num_imgs, img.rows)));
        if (!cnn_rgb) {
            cv::Mat img_gray;
            cv::cvtColor(imgs[i], img_gray, cv::COLOR_BGR2GRAY);
            imgs_gray.emplace_back(img_gray);
        }
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
        std::pair<cv::Mat, cv::Mat> ret;
        if (cnn_rgb) {
            ret = stereo->estimatePointsViaRaw(imgs[stereo->cam_idx_a], imgs[stereo->cam_idx_b], cv::Mat(), show);
        } else {
            ret = stereo->estimatePointsViaRaw(imgs_gray[stereo->cam_idx_a], imgs_gray[stereo->cam_idx_b], imgs[stereo->cam_idx_a], show);
        }
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

cv::Mat readVingette(const std::string & mask_file, double avg_brightness) {
    cv::Mat photometric_inv;
    cv::Mat photometric_calib = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);
    std::cout << photometric_calib.type() << std::endl;
    if (photometric_calib.type() == CV_8U) {
        photometric_calib.convertTo(photometric_calib, CV_32FC1, 1.0/255.0);
    } else if (photometric_calib.type() == CV_16S) {
        photometric_calib.convertTo(photometric_calib, CV_32FC1, 1.0/65535.0);
    }
    cv::divide(avg_brightness, photometric_calib, photometric_inv);
    return photometric_inv;
}

void QuadCamDepthEst::loadCameraConfig(YAML::Node & config, std::string configPath) {
    int _camera_config = config["camera_configuration"].as<int>();
    camera_config = (CameraConfig)_camera_config;

    cv::Mat photometric_inv_vec[4];
    cv::Mat photometric_inv, photometric_inv_1;
    float avg_brightness = config["avg_brightness"].as<float>();
    if (config["photometric_calib"]) {
        photometric_inv = readVingette(configPath + "/" + config["photometric_calib"].as<std::string>(), avg_brightness);
    }
    if (config["photometric_calib_1"]) {
        photometric_inv_1 = readVingette(configPath + "/" + config["photometric_calib_1"].as<std::string>(), avg_brightness);
    }
    int num_cemaras = config["photometric_calib_numbers"].as<int>();
    if (num_cemaras >0 && num_cemaras <= 4){
        std::string photometric_calib_path = config["photometric_calib_path"].as<std::string>();
        if( access(photometric_calib_path.c_str(),F_OK) == -1){
            printf("[QuadCamDepthEst] photometric_calib_path is not exist use defualt image\n");
            for (int i = 0; i < num_cemaras; i++) {
                photometric_inv_vec[i] = cv::Mat (1280, 720, CV_8UC3, cv::Scalar(255, 255, 255));
            }
            printf("[DEBUG] use pure white\n");
        } else {
            for(int i = 0 ; i< num_cemaras; i++){
                std::string mask_file = photometric_calib_path + "/" + std::string("cam_") + std::to_string(i) + std::string("_vig_mask.png");//search image "cam_i_vig_mask.png"
                photometric_inv_vec[i] = readVingette(mask_file, avg_brightness);
                printf("[Read vingette]:%s\n",mask_file.c_str());
            }
        }
    }

    std::string calib_file_path = config["cam_calib_file_path"].as<std::string>();
    printf("[QuadCamDepthEst] Load camera config from %s\n", calib_file_path.c_str());
    // calib_file_path = configPath + "/" + calib_file_path;
    YAML::Node config_cams = YAML::LoadFile(calib_file_path); 
    //read intrinsic and distortion model

    int32_t photometric_inv_idx = 0;
    for (const auto& kv : config_cams) {
        std::string camera_name = kv.first.as<std::string>();
        printf("[QuadCamDepthEst] Load camera %s\n", camera_name.c_str());
        const YAML::Node& value = kv.second;
        auto ret = D2FrontEnd::readCameraConfig(camera_name, value);
        raw_cameras.emplace_back(ret.first);
        if (camera_config == CameraConfig::FOURCORNER_FISHEYE) {
            double fov = config["fov"].as<double>();
            if(photometric_inv_idx >=4 || photometric_inv_idx < 0){
                photometric_inv_idx = 0;
            }
            printf("[Debug ]undistortor matrix init with size width:%d height:%d\n",this->width,this->height);
            undistortors.push_back(new D2Common::FisheyeUndist(ret.first, 0, fov, true,
                D2Common::FisheyeUndist::UndistortPinhole2, this->width, this->height, photometric_inv_vec[photometric_inv_idx]));
            printf("[Debug] undistorter width and height:%d %d\n",this->width,this->height);
            photometric_inv_idx++;
        }
        raw_cam_extrinsics.emplace_back(ret.second);
    }
    
    if (camera_config == CameraConfig::STEREO_PINHOLE) {
        //In stereo mode
        Matrix4d T;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                T(i, j) = config_cams["cam1"]["T_cn_cnm1"][i][j].as<double>();
            }
        }
        auto baseline = Swarm::Pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
        auto stereo = new VirtualStereo(baseline, raw_cameras[0], raw_cameras[1], hitnet, crestereo);
        stereo->extrinsic = raw_cam_extrinsics[0];
        stereo->enable_texture = enable_texture;
        stereo->initVingette(photometric_inv, photometric_inv_1);
        virtual_stereos.emplace_back(stereo);
    } else if (camera_config == CameraConfig::FOURCORNER_FISHEYE) {
        printf("Config fourfisheye\n");
        for (const auto & kv: config["stereos"]) {
            auto node = kv.second;
            std::string stereo_name = kv.first.as<std::string>();
            int cam_idx_l = node["cam_idx_l"].as<int>();
            int cam_idx_r = node["cam_idx_r"].as<int>();
            int idx_l = node["idx_l"].as<int>();
            int idx_r = node["idx_r"].as<int>();
            std::string stereo_calib_file = node["stereo_config"].as<std::string>();
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
                undistortors[cam_idx_l], undistortors[cam_idx_r], idx_l, idx_r, hitnet, crestereo);
            auto att = undistortors[cam_idx_l]->t[idx_l];
            stereo->extrinsic = raw_cam_extrinsics[cam_idx_l] * Swarm::Pose(att, Vector3d(0, 0, 0));
            stereo->enable_texture = enable_texture;
            stereo->initRecitfy(baseline, KD0.first, KD0.second, KD1.first, KD1.second);
            virtual_stereos.emplace_back(stereo);
        }
    }
}

std::pair<cv::Mat, cv::Mat> QuadCamDepthEst::intrinsicsFromNode(const YAML::Node & node) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
    printf("calibration parameters in size  height:%d width:%d\n",node["resolution"][1].as<int>(),node["resolution"][0].as<int>());

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

