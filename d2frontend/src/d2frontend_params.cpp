#include "d2frontend/d2frontend_params.h"
#include "d2frontend/loop_cam.h"
#include "d2frontend/loop_detector.h"
#include "d2frontend/d2featuretracker.h"
#include "swarm_msgs/swarm_lcm_converter.hpp"
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>
#include <camodocal/camera_models/CataCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>
#include <d2common/fisheye_undistort.h>

namespace D2FrontEnd {
    D2FrontendParams * params;
    std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config);
    D2FrontendParams::D2FrontendParams(ros::NodeHandle & nh)
    {
        //Read VINS params.
        nh.param<std::string>("vins_config_path",vins_config_path, "");
        cv::FileStorage fsSettings;
        fsSettings.open(vins_config_path.c_str(), cv::FileStorage::READ);
        int pn = vins_config_path.find_last_of('/');
        std::string configPath = vins_config_path.substr(0, pn);

        loopcamconfig = new LoopCamConfig;
        loopdetectorconfig = new LoopDetectorConfig;
        ftconfig = new D2FTConfig;

        //Basic confi
        nh.param<int>("self_id", self_id, -1);
        int _camconfig = fsSettings["camera_configuration"];
        camera_configuration = (CameraConfig) _camconfig;
        nh.param<double>("nonkeyframe_waitsec", ACCEPT_NONKEYFRAME_WAITSEC, 5.0);
        nh.param<double>("min_movement_keyframe", min_movement_keyframe, 0.3);
        estimation_mode = (ESTIMATION_MODE) (int) fsSettings["estimation_mode"];

        //Debug configs
        nh.param<bool>("send_img", send_img, false);
        nh.param<int>("jpg_quality", JPG_QUALITY, 50);
        nh.param<bool>("send_whole_img_desc", send_whole_img_desc, false);
        nh.param<bool>("show", show, false);
        nh.param<bool>("debug_no_rejection", loopdetectorconfig->DEBUG_NO_REJECT, false);
        nh.param<bool>("enable_pub_remote_frame", enable_pub_remote_frame, false);
        nh.param<bool>("enable_pub_local_frame", enable_pub_local_frame, false);
        nh.param<bool>("enable_sub_remote_frame", enable_sub_remote_frame, false);
        nh.param<std::string>("output_path", OUTPUT_PATH, "");
        
        enable_perf_output = (int) fsSettings["enable_perf_output"];
        enbale_detailed_output = (int) fsSettings["enbale_detailed_output"];
        enbale_speed_ouptut = (int) fsSettings["enbale_speed_ouptut"];

        print_network_status = (int) fsSettings["print_network_status"];
        verbose = (int) fsSettings["verbose"];
        ftconfig->write_to_file = (int) fsSettings["write_tracking_image_to_file"];

        //
        use_gpu_feature_tracking = (int)fsSettings["use_gpu_feature_tracking"];
        use_gpu_good_feature_extraction = (int)fsSettings["use_gpu_good_feature_extraction"];

        //Loopcam configs
        total_feature_num = (int) fsSettings["max_cnt"];
        loopcamconfig->superpoint_max_num = (int) fsSettings["max_superpoint_cnt"];
        loopcamconfig->DEPTH_FAR_THRES = fsSettings["depth_far_thres"];
        loopcamconfig->DEPTH_NEAR_THRES = fsSettings["depth_near_thres"];
        loopcamconfig->show = (int) fsSettings["show_feature_extraction"];
        enable_pca_superpoint = (int) fsSettings["enable_pca_superpoint"];
        enable_pca_netvlad = (int) fsSettings["enable_pca_netvlad"];
        loopdetectorconfig->loop_detection_netvlad_thres = fsSettings["loop_detection_netvlad_thres"];
        track_remote_netvlad_thres = fsSettings["track_remote_netvlad_thres"]; //This is for d2featuretracker
        if (enable_pca_superpoint) {
            superpoint_dims = (int) fsSettings["superpoint_pca_dims"];
        }
        if (enable_pca_netvlad) {
            netvlad_dims = (int) fsSettings["netvlad_pca_dims"];
            track_remote_netvlad_thres = 1.46 * track_remote_netvlad_thres - 0.499128; //This is computed in pca_decomp.ipynb
            loopdetectorconfig->loop_detection_netvlad_thres = 1.46 * loopdetectorconfig->loop_detection_netvlad_thres - 0.499128;
        }
        nh.param<double>("superpoint_thres", loopcamconfig->superpoint_thres, 0.012);
        nh.param<std::string>("pca_comp_path",loopcamconfig->pca_comp, "");
        nh.param<std::string>("pca_mean_path",loopcamconfig->pca_mean, "");
        nh.param<std::string>("pca_netvlad", pca_netvlad, "");
        nh.param<std::string>("superpoint_model_path", loopcamconfig->superpoint_model, "");
        nh.param<std::string>("netvlad_model_path", loopcamconfig->netvlad_model, "");
        loopcamconfig->cnn_enable_tensorrt = (int) fsSettings["cnn_enable_tensorrt"];
        loopcamconfig->cnn_enable_tensorrt_int8 = (int) fsSettings["cnn_enable_tensorrt_int8"];
        if (loopcamconfig->cnn_enable_tensorrt_int8) {
            loopcamconfig->netvlad_int8_calib_table_name = (std::string) fsSettings["netvlad_int8_calib_table_name"];
            loopcamconfig->superpoint_int8_calib_table_name = (std::string) fsSettings["superpoint_int8_calib_table_name"];
        }
        
        //NN engine type
        nh.param<int>("nn_engine_type", (int &) loopcamconfig->nn_engine_type, 0);
        nh.param<bool>("lower_cam_as_main", loopcamconfig->right_cam_as_main, false);
        nh.param<double>("triangle_thres", loopcamconfig->TRIANGLE_THRES, 0.006);
        nh.param<bool>("output_raw_superpoint_desc", loopcamconfig->OUTPUT_RAW_SUPERPOINT_DESC, false);
        nh.param<int>("accept_min_3d_pts", loopcamconfig->ACCEPT_MIN_3D_PTS, 50);
        loopcamconfig->camera_configuration = camera_configuration;
        loopcamconfig->self_id = self_id;
        loopcamconfig->cnn_use_onnx = (int) fsSettings["cnn_use_onnx"];
        loopcamconfig->send_img = send_img;
        loopcamconfig->superpoint_trt_engine_path = (std::string) fsSettings["superpoint_trt_path"];
        loopcamconfig->netvlad_trt_engine_path = (std::string) fsSettings["moblieNetVlad_trt_path"];
        loopcamconfig->nn_engine_type = (EngineType) (int) fsSettings["nn_engine_type"];
        //Feature tracker.        
        ftconfig->show_feature_id = (int) fsSettings["show_track_id"];
        ftconfig->long_track_frames = fsSettings["landmark_estimate_tracks"];
        ftconfig->check_essential = (int) fsSettings["check_essential"];
        ftconfig->enable_lk_optical_flow = (int) fsSettings["enable_lk_optical_flow"];
        ftconfig->lk_use_fast = (int) fsSettings["lk_use_fast"];
        ftconfig->remote_min_match_num = fsSettings["remote_min_match_num"];
        ftconfig->double_counting_common_feature = (int) fsSettings["double_counting_common_feature"];
        ftconfig->enable_superglue_local = (int) fsSettings["enable_superglue_local"];
        ftconfig->enable_superglue_remote = (int) fsSettings["enable_superglue_remote"];
        ftconfig->ransacReprojThreshold = fsSettings["ransacReprojThreshold"];
        ftconfig->parallex_thres = fsSettings["parallex_thres"];
        ftconfig->knn_match_ratio = fsSettings["knn_match_ratio"];
        ftconfig->frame_step = fsSettings["frame_step"];
        nh.param<int>("long_track_thres", ftconfig->long_track_thres, 0);
        nh.param<int>("last_track_thres", ftconfig->last_track_thres, 20);
        nh.param<double>("new_feature_thres", ftconfig->new_feature_thres, 0.5);
        nh.param<int>("min_keyframe_num", ftconfig->min_keyframe_num, 2);
        nh.param<std::string>("superglue_model_path", ftconfig->superglue_model_path, "");
        ftconfig->enable_search_local_aera_remote = (int) fsSettings["enable_search_local_aera_remote"];
        ftconfig->enable_motion_prediction_local = (int) fsSettings["enable_motion_prediction_local"];
        if (!fsSettings["enable_search_local_aera"].empty()) {
            ftconfig->enable_search_local_aera = (int) fsSettings["enable_search_local_aera"];
            ftconfig->search_local_max_dist = fsSettings["search_local_max_dist"];
        } else {
            printf("[D2FrontendParams] enable_search_local_aera not found, use default\n");
        }
        if (!fsSettings["feature_min_dist"].empty()) {
            feature_min_dist = fsSettings["feature_min_dist"];
        } else {
            printf("[D2FrontendParams] feature_min_dist not found, use default\n");
        }
        ftconfig->track_from_keyframe = (int) fsSettings["track_from_keyframe"];
        //Loop detector
        loopdetectorconfig->enable_homography_test = (int) fsSettings["enable_homography_test"];
        loopdetectorconfig->accept_loop_max_yaw = (double) fsSettings["accept_loop_max_yaw"];
        loopdetectorconfig->accept_loop_max_pos = (double) fsSettings["accept_loop_max_pos"];
        loopdetectorconfig->loop_inlier_feature_num = fsSettings["loop_inlier_feature_num"];
        loopdetectorconfig->knn_match_ratio = fsSettings["knn_match_ratio"];
        loopdetectorconfig->gravity_check_thres = fsSettings["gravity_check_thres"];
        nh.param<bool>("enable_loop", enable_loop, true);
        nh.param<bool>("is_4dof", loopdetectorconfig->is_4dof, true);
        nh.param<int>("match_index_dist", loopdetectorconfig->match_index_dist, 10);
        nh.param<int>("match_index_dist_remote", loopdetectorconfig->match_index_dist_remote, 10);
        nh.param<int>("min_match_per_dir", loopdetectorconfig->MIN_MATCH_PRE_DIR, 15);
        nh.param<int>("inter_drone_init_frames", loopdetectorconfig->inter_drone_init_frames, 50);
        nh.param<double>("detector_match_thres", loopdetectorconfig->DETECTOR_MATCH_THRES, 0.9);
        nh.param<double>("odometry_consistency_threshold", loopdetectorconfig->odometry_consistency_threshold, 2.0);
        nh.param<double>("pos_covariance_per_meter", loopdetectorconfig->pos_covariance_per_meter, 0.01);
        nh.param<double>("yaw_covariance_per_meter", loopdetectorconfig->yaw_covariance_per_meter, 0.003);
        nh.param<double>("loop_cov_pos", loopdetectorconfig->loop_cov_pos, 0.013);
        nh.param<double>("loop_cov_ang", loopdetectorconfig->loop_cov_ang, 2.5e-04);
        nh.param<int>("min_direction_loop", loopdetectorconfig->MIN_DIRECTION_LOOP, 3);
        pgo_mode = static_cast<PGO_MODE>((int) fsSettings["pgo_mode"]);
        nh.param<std::string>("superglue_model_path", loopdetectorconfig->superglue_model_path, "");

        //Network config
        nh.param<std::string>("lcm_uri", _lcm_uri, "udpm://224.0.0.251:7667?ttl=1");
        nh.param<double>("recv_msg_duration", recv_msg_duration, 0.5);
        nh.param<bool>("enable_network", enable_network, true);
        lazy_broadcast_keyframe = (int) fsSettings["lazy_broadcast_keyframe"];
        printf("[D2Frontend] Using lazy broadcast keyframe: %d\n", lazy_broadcast_keyframe);

        if (camera_configuration == CameraConfig::STEREO_PINHOLE) {
            loopdetectorconfig->MAX_DIRS = 1;
            min_receive_images = 2;
        } else if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
            loopdetectorconfig->MAX_DIRS = 4;
            min_receive_images = 2;
        } else if (camera_configuration == CameraConfig::PINHOLE_DEPTH) {
            loopdetectorconfig->MAX_DIRS = 1;
            min_receive_images = 1;
        } else if (camera_configuration == CameraConfig::FOURCORNER_FISHEYE)  {
            loopdetectorconfig->MAX_DIRS = 4;
            min_receive_images = 4;
        } else {
            ROS_ERROR("[D2Frontend] Camera configuration %d not implement yet.", camera_configuration);
            exit(-1);
        }
        depth_topics.emplace_back((std::string) fsSettings["depth_topic"]);
        is_comp_images = (int) fsSettings["is_compressed_images"];
        generateCameraModels(fsSettings, configPath);
    }

    void D2FrontendParams::generateCameraModels(cv::FileStorage & fsSettings, std::string configPath) {
        readCameraConfigs(fsSettings, configPath);
        camodocal::CameraFactory cam_factory;
        for (auto & cam_calib_path : camera_config_paths) {
            ROS_INFO("[D2Frontend] Read camera from %s", cam_calib_path.c_str());
            auto cam = cam_factory.generateCameraFromYamlFile(cam_calib_path);
            if (cam) {
                camera_ptrs.push_back(cam);
            } else {
                ROS_ERROR("[D2Frontend]Failed to read camera from %s", cam_calib_path.c_str());
            }
        }
        //TODO::Multi calibration results
        std::string photometric_calib_file = fsSettings["photometric_calib"];
        cv::Mat photometric;
        if ( photometric_calib_file != "") {
            double avg_photometric = fsSettings["avg_photometric"];
            photometric = cv::imread(configPath + "/" + photometric_calib_file, cv::IMREAD_GRAYSCALE);
            photometric.convertTo(photometric, CV_32FC1, 1.0/255.0);
            cv::divide(avg_photometric, photometric, photometric);
            printf("[D2Frontend] Read photometric calibration from: %s\n", photometric_calib_file.c_str());
        } else {
            printf("[D2Frontend] No photometric calibration file provided.\n");
        }
        if (enable_undistort_image) {
            raw_camera_ptrs = camera_ptrs;
            camera_ptrs.clear();
            for (auto cam: raw_camera_ptrs) { 
                auto ptr = new FisheyeUndist(cam, 0, undistort_fov, true, FisheyeUndist::UndistortCylindrical, 
                    width_undistort, height_undistort, photometric);
                auto cylind_cam = ptr->cam_top;
                camera_ptrs.push_back(cylind_cam);
                undistortors.emplace_back(ptr);
                // focal_length = static_cast<camodocal::CylindricalCamera * >(cylind_cam.get())->getParameters().fx();
            }
        } else {
            // auto cam = camera_ptrs[0];
            // if (cam->modelType() == camodocal::Camera::MEI) {
            //     focal_length = static_cast<camodocal::CataCamera* >(cam.get())->getParameters().gamma1();
            // } else if (cam->modelType() == camodocal::Camera::PINHOLE) {
            //     focal_length = static_cast<camodocal::PinholeCamera* >(cam.get())->getParameters().fx();
            // }
        }
        printf("[D2Frontend] Focal length initialize to: %.1f\n", focal_length);
    }


    void D2FrontendParams::readCameraConfigs(cv::FileStorage & fsSettings, std::string configPath) {
        enable_undistort_image = loopcamconfig->enable_undistort_image = (int) fsSettings["enable_undistort_image"];
        width_undistort = (int) fsSettings["width_undistort"];
        height_undistort = (int) fsSettings["height_undistort"];
        image_frequency = (int) fsSettings["image_freq"];
        undistort_fov = fsSettings["undistort_fov"];
        width = (int) fsSettings["image_width"];
        height = (int) fsSettings["image_height"];
        std::string camera_seq_str = fsSettings["camera_seq"]; // Back-right Back-left Front-left Front-right
        if (camera_seq_str == "") {
            this->camera_seq = std::vector<int>{0, 1, 2, 3};
        } else {
            //Camera seq from string "0123" to vector<int> {0, 1, 2, 3}
            this->camera_seq = std::vector<int>(camera_seq_str.size());
            std::transform(camera_seq_str.begin(), camera_seq_str.end(), this->camera_seq.begin(), [](char c) { return c - '0'; });
        }
        
        std::string calib_file_path = fsSettings["calib_file_path"];
        if (calib_file_path != "") {
            calib_file_path = configPath + "/" + calib_file_path;
            ROS_INFO("Will read camera calibration from %s", calib_file_path.c_str());
            readCameraCalibrationfromFile(calib_file_path);
            int camera_num = extrinsics.size();
            for (auto i = 0; i < camera_num; i++) {
                char param_name[64] = {0};
                sprintf(param_name, "image%d_topic", i);
                std::string topic = (std::string)  fsSettings[param_name];
                image_topics.emplace_back(topic);
            }
        } else {
            ROS_INFO("Read camera from config file");
            //Camera configurations from VINS config.
            int camera_num = fsSettings["num_of_cam"];
            for (auto i = 0; i < camera_num; i++) {
                std::string index = std::to_string(i);
                char param_name[64] = {0};
                sprintf(param_name, "image%d_topic", i);
                std::string topic = (std::string)  fsSettings[param_name];

                sprintf(param_name, "cam%d_calib", i);
                std::string camera_calib_path = (std::string) fsSettings[param_name];
                camera_calib_path = configPath + "/" + camera_calib_path;

                sprintf(param_name, "body_T_cam%d", i);
                cv::Mat cv_T;
                fsSettings[param_name] >> cv_T;
                Eigen::Matrix4d T;
                cv::cv2eigen(cv_T, T);
                Swarm::Pose pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
                
                image_topics.emplace_back(topic);
                camera_config_paths.emplace_back(camera_calib_path);
                extrinsics.emplace_back(pose);

                ROS_INFO("[SWARM_LOOP] Camera %d: topic: %s, calib: %s, T: %s", 
                    i, topic.c_str(), camera_calib_path.c_str(), pose.toStr().c_str());
            }
        }
    }

    void D2FrontendParams::readCameraCalibrationfromFile(const std::string & path) {
        YAML::Node config = YAML::LoadFile(path);
        for (const auto& kv : config) {
                std::string camera_name = kv.first.as<std::string>();
                std::cout << camera_name << "\n";
                const YAML::Node& value = kv.second;
                auto ret = readCameraConfig(camera_name, value);
                camera_ptrs.emplace_back(ret.first);
                extrinsics.emplace_back(ret.second);
        }
    }

    std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config) {
        //In this case, we generate camera ptr.
        //Now only accept omni-radtan.
        camodocal::CameraPtr camera;
        if (config["camera_model"].as<std::string>() == "omni" && 
            config["distortion_model"].as<std::string>() == "radtan" ) {
            int width = config["resolution"][0].as<int>();
            int height = config["resolution"][1].as<int>();
            double xi = config["intrinsics"][0].as<double>();
            double gamma1 = config["intrinsics"][1].as<double>();
            double gamma2 = config["intrinsics"][2].as<double>();
            double u0 = config["intrinsics"][3].as<double>();
            double v0 = config["intrinsics"][4].as<double>();
            double k1 = config["distortion_coeffs"][0].as<double>();
            double k2 = config["distortion_coeffs"][1].as<double>();
            double p1 = config["distortion_coeffs"][2].as<double>();
            double p2 = config["distortion_coeffs"][3].as<double>();
            printf("Camera %s model omni-radtan\n width: %d, height: %d, xi: %f, gamma1: %f, gamma2: %f, u0: %f, v0: %f, k1: %f, k2: %f, p1: %f, p2: %f\n", 
                camera_name.c_str(), width, height, xi, gamma1, gamma2, u0, v0, k1, k2, p1, p2);
            camera = camodocal::CataCameraPtr(new camodocal::CataCamera(camera_name,
               width, height, xi, k1, k2, p1, p2,
               gamma1, gamma2, u0, v0));
        }   
        else if (config["camera_model"].as<std::string>() == "pinhole" && 
            config["distortion_model"].as<std::string>() == "radtan" ) {
            int width = config["resolution"][0].as<int>();
            int height = config["resolution"][1].as<int>();
            double fx = config["intrinsics"][0].as<double>();
            double fy = config["intrinsics"][1].as<double>();
            double cx = config["intrinsics"][2].as<double>();
            double cy = config["intrinsics"][3].as<double>();

            double k1 = config["distortion_coeffs"][0].as<double>();
            double k2 = config["distortion_coeffs"][1].as<double>();
            double p1 = config["distortion_coeffs"][2].as<double>();
            double p2 = config["distortion_coeffs"][3].as<double>();
            printf("Camera %s model pinhole-radtan\n width: %d, height: %d, fx: %f, fy: %f, cx: %f, cy: %f, k1: %f, k2: %f, p1: %f, p2: %f\n", 
                camera_name.c_str(), width, height, fx, fy, cx, cy, k1, k2, p1, p2);
            camera = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera(camera_name,
               width, height, k1, k2, p1, p2, fx, fy, cx, cy));
        }
        else {
            printf("Camera not supported yet, please fillin in src/d2frontend_params.cpp function: readCameraConfig\n");
            exit(-1);
        }
        Matrix4d T;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                T(i, j) = config["T_cam_imu"][i][j].as<double>();
            }
        }
        Matrix3d R = T.block<3, 3>(0, 0).transpose();
        Vector3d t = -R*T.block<3, 1>(0, 3);
        #if 0 //xuhao version
        Swarm::Pose pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
        std::cout << "T_cam_imu:\n" << T << std::endl;
        std::cout << "pose:\n" << pose.toStr() << std::endl;
        #endif
        Swarm::Pose pose(R, t);
        std::cout <<"T_cam_imu:\n" << R << std::endl;
        std::cout << "pose:\n" << pose.toStr() << std::endl;
        
        return std::make_pair(camera, pose);
    }

}