#include "d2frontend/d2frontend_params.h"
#include "d2frontend/loop_cam.h"
#include "d2frontend/loop_detector.h"
#include "d2frontend/d2featuretracker.h"
#include "swarm_msgs/swarm_lcm_converter.hpp"
#include <opencv2/core/eigen.hpp>

namespace D2FrontEnd {
    D2FrontendParams * params;
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
        nh.param<double>("max_freq", max_freq, 1.0);
        nh.param<double>("nonkeyframe_waitsec", ACCEPT_NONKEYFRAME_WAITSEC, 5.0);
        nh.param<double>("min_movement_keyframe", min_movement_keyframe, 0.3);

        //Debug configs
        nh.param<bool>("send_img", send_img, false);
        nh.param<int>("jpg_quality", JPG_QUALITY, 50);
        nh.param<bool>("send_whole_img_desc", send_whole_img_desc, false);
        nh.param<bool>("debug_image", debug_image, false);
        nh.param<bool>("debug_no_rejection", loopdetectorconfig->DEBUG_NO_REJECT, false);
        nh.param<bool>("enable_pub_remote_frame", enable_pub_remote_frame, false);
        nh.param<bool>("enable_pub_local_frame", enable_pub_local_frame, false);
        nh.param<bool>("enable_sub_remote_frame", enable_sub_remote_frame, false);
        nh.param<bool>("verbose", verbose, false);
        nh.param<std::string>("output_path", OUTPUT_PATH, "");
        enable_perf_output = (int) fsSettings["enable_perf_output"];

        //Loopcam configs
        loopcamconfig->width = (int) fsSettings["image_width"];
        loopcamconfig->height = (int) fsSettings["image_height"];
        loopcamconfig->superpoint_max_num = (int) fsSettings["max_superpoint_cnt"];
        total_feature_num = (int) fsSettings["max_cnt"];
        loopcamconfig->DEPTH_FAR_THRES = fsSettings["depth_far_thres"];
        loopcamconfig->DEPTH_NEAR_THRES = fsSettings["depth_near_thres"];
        nh.param<double>("superpoint_thres", loopcamconfig->superpoint_thres, 0.012);
        nh.param<std::string>("pca_comp_path",loopcamconfig->pca_comp, "");
        nh.param<std::string>("pca_mean_path",loopcamconfig->pca_mean, "");
        nh.param<std::string>("superpoint_model_path", loopcamconfig->superpoint_model, "");
        nh.param<std::string>("netvlad_model_path", loopcamconfig->netvlad_model, "");
        nh.param<bool>("lower_cam_as_main", loopcamconfig->right_cam_as_main, false);
        nh.param<double>("triangle_thres", loopcamconfig->TRIANGLE_THRES, 0.006);
        nh.param<bool>("output_raw_superpoint_desc", loopcamconfig->OUTPUT_RAW_SUPERPOINT_DESC, false);
        nh.param<int>("accept_min_3d_pts", loopcamconfig->ACCEPT_MIN_3D_PTS, 50);
        loopcamconfig->camera_configuration = camera_configuration;
        loopcamconfig->self_id = self_id;
        loopcamconfig->cnn_use_onnx = (int) fsSettings["cnn_use_onnx"];
        loopcamconfig->send_img = send_img;
        loopcamconfig->show = debug_image;

        //Feature tracker.
        ftconfig->show_feature_id = (int) fsSettings["show_track_id"];
        ftconfig->long_track_frames = fsSettings["landmark_estimate_tracks"];
        ftconfig->check_homography = (int) fsSettings["check_homography"];
        ftconfig->enable_lk_optical_flow = (int) fsSettings["enable_lk_optical_flow"];
        vlad_threshold = fsSettings["vlad_threshold"];
        nh.param<int>("long_track_thres", ftconfig->long_track_thres, 20);
        nh.param<int>("last_track_thres", ftconfig->last_track_thres, 20);
        nh.param<double>("new_feature_thres", ftconfig->new_feature_thres, 0.5);
        nh.param<double>("parallex_thres", ftconfig->parallex_thres, 10.0/460.0);
        nh.param<int>("min_keyframe_num", ftconfig->min_keyframe_num, 2);
        nh.param<double>("ransacReprojThreshold", ftconfig->ransacReprojThreshold, 10.0);

        //Loop detector
        nh.param<bool>("enable_loop", enable_loop, true);
        nh.param<bool>("is_4dof", loopdetectorconfig->is_4dof, true);
        nh.param<int>("init_loop_min_feature_num", loopdetectorconfig->INIT_MODE_MIN_LOOP_NUM, 10);
        nh.param<int>("match_index_dist", loopdetectorconfig->MATCH_INDEX_DIST, 10);
        nh.param<int>("min_loop_feature_num", loopdetectorconfig->MIN_LOOP_NUM, 15);
        nh.param<int>("min_match_per_dir", loopdetectorconfig->MIN_MATCH_PRE_DIR, 15);
        nh.param<int>("inter_drone_init_frames", loopdetectorconfig->inter_drone_init_frames, 50);
        nh.param<double>("query_thres", loopdetectorconfig->INNER_PRODUCT_THRES, 0.6);
        nh.param<double>("init_query_thres", loopdetectorconfig->INIT_MODE_PRODUCT_THRES, 0.3);
        nh.param<double>("detector_match_thres", loopdetectorconfig->DETECTOR_MATCH_THRES, 0.9);
        nh.param<double>("odometry_consistency_threshold", loopdetectorconfig->odometry_consistency_threshold, 2.0);
        nh.param<double>("pos_covariance_per_meter", loopdetectorconfig->pos_covariance_per_meter, 0.01);
        nh.param<double>("yaw_covariance_per_meter", loopdetectorconfig->yaw_covariance_per_meter, 0.003);
        nh.param<double>("loop_cov_pos", loopdetectorconfig->loop_cov_pos, 0.013);
        nh.param<double>("loop_cov_ang", loopdetectorconfig->loop_cov_ang, 2.5e-04);
        nh.param<int>("min_direction_loop", loopdetectorconfig->MIN_DIRECTION_LOOP, 3);

        //Network config
        nh.param<std::string>("lcm_uri", _lcm_uri, "udpm://224.0.0.251:7667?ttl=1");
        nh.param<bool>("send_all_features", SEND_ALL_FEATURES, false);
        nh.param<double>("recv_msg_duration", recv_msg_duration, 0.5);
        nh.param<bool>("enable_network", enable_network, true);

        if (camera_configuration == CameraConfig::STEREO_PINHOLE) {
            loopdetectorconfig->MAX_DIRS = 1;
        } else if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
            loopdetectorconfig->MAX_DIRS = 4;
        } else if (camera_configuration == CameraConfig::PINHOLE_DEPTH) {
            loopdetectorconfig->MAX_DIRS = 1;
        } else {
            loopdetectorconfig->MAX_DIRS = 0;
            ROS_ERROR("[SWARM_LOOP] Camera configuration %d not implement yet.", camera_configuration);
            exit(-1);
        }

        //Camera configurations from VINS config.
        int camera_num = fsSettings["num_of_cam"];
        for (auto i = 0; i < camera_num; i++) {
            std::string index = std::to_string(i);
            char param_name[64] = {0};
            sprintf(param_name, "image%d_topic", i);
            std::string topic = (std::string)  fsSettings[param_name];

            sprintf(param_name, "compressed_image%d_topic", i);
            std::string comp_topic = (std::string) fsSettings[param_name];

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
            comp_image_topics.emplace_back(comp_topic);
            loopcamconfig->camera_config_paths.emplace_back(camera_calib_path);
            extrinsics.emplace_back(pose);

            ROS_INFO("[SWARM_LOOP] Camera %d: topic: %s, comp_topic: %s, calib: %s, T: %s", 
                i, topic.c_str(), comp_topic.c_str(), camera_calib_path.c_str(), pose.toStr().c_str());
        }

        depth_topics.emplace_back((std::string) fsSettings["depth_topic"]);
        is_comp_images = (int) fsSettings["is_compressed_images"];

    }
}