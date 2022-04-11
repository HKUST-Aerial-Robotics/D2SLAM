#include "d2frontend/d2frontend_params.h"
#include "loop_cam.h"
namespace D2Frontend {
    D2FrontendParams::D2FrontendParams(ros::NodeHandle & nh)
    {
        //Basic confi
        nh.param<int>("self_id", self_id, -1);
        int _camconfig;
        nh.param<int>("camera_configuration", _camconfig, 1);
        camera_configuration = (CameraConfig) _camconfig;
        nh.param<double>("max_freq", max_freq, 1.0);

        //Debug configs
        nh.param<bool>("send_img", send_img, false);
        nh.param<int>("jpg_quality", JPG_QUALITY, 50);
        nh.param<bool>("is_pc_replay", IS_PC_REPLAY, false);
        nh.param<bool>("send_whole_img_desc", send_whole_img_desc, false);
        nh.param<bool>("debug_image", debug_image, false);
        nh.param<bool>("debug_no_rejection", DEBUG_NO_REJECT, false);
        
        //Loopcam configs
        nh.param<double>("superpoint_thres", loopcamconfig.superpoint_thres, 0.012);
        nh.param<int>("superpoint_max_num", loopcamconfig.superpoint_max_num, 200);
        nh.param<std::string>("pca_comp_path",loopcamconfig._pca_comp, "");
        nh.param<std::string>("pca_mean_path",loopcamconfig._pca_mean_path, "");
        nh.param<std::string>("superpoint_model_path", loopcamconfig.superpoint_model, "");
        nh.param<std::string>("netvlad_model_path", loopcamconfig.netvlad_model, "");
        nh.param<std::string>("camera_config_path", loopcamconfig.camera_config_path, 
            "/home/xuhao/swarm_ws/src/VINS-Fusion-gpu/config/vi_car/cam0_mei.yaml");
        nh.param<int>("width", loopcamconfig.width, 400);
        nh.param<int>("height", loopcamconfig.height, 208);       
        nh.param<std::string>("vins_config_path",vins_config_path, "");
        nh.param<bool>("lower_cam_as_main", LOWER_CAM_AS_MAIN, false);
        nh.param<double>("triangle_thres", TRIANGLE_THRES, 0.006);
        nh.param<double>("depth_far_thres", DEPTH_FAR_THRES, 10.0);
        nh.param<double>("depth_near_thres", DEPTH_NEAR_THRES, 0.3);

        loopcamconfig.camera_configuration = camera_configuration;
        loopcamconfig.self_id = self_id;

        //Loop detector
        nh.param<bool>("is_4dof", is_4dof, true);
        nh.param<double>("min_movement_keyframe", min_movement_keyframe, 0.3);
        nh.param<double>("nonkeyframe_waitsec", ACCEPT_NONKEYFRAME_WAITSEC, 5.0);

        nh.param<int>("init_loop_min_feature_num", INIT_MODE_MIN_LOOP_NUM, 10);
        nh.param<int>("match_index_dist", MATCH_INDEX_DIST, 10);
        nh.param<int>("min_loop_feature_num", MIN_LOOP_NUM, 15);
        nh.param<int>("min_match_per_dir", MIN_MATCH_PRE_DIR, 15);
        nh.param<int>("accept_min_3d_pts", ACCEPT_MIN_3D_PTS, 50);
        nh.param<int>("inter_drone_init_frames", inter_drone_init_frames, 50);
        nh.param<bool>("enable_pub_remote_frame", enable_pub_remote_frame, false);
        nh.param<bool>("enable_pub_local_frame", enable_pub_local_frame, false);
        nh.param<bool>("enable_sub_remote_frame", enable_sub_remote_frame, false);

        nh.param<double>("query_thres", INNER_PRODUCT_THRES, 0.6);
        nh.param<double>("init_query_thres", INIT_MODE_PRODUCT_THRES, 0.3);
        nh.param<double>("recv_msg_duration", recv_msg_duration, 0.5);
        nh.param<double>("detector_match_thres", DETECTOR_MATCH_THRES, 0.9);
        nh.param<bool>("output_raw_superpoint_desc", OUTPUT_RAW_SUPERPOINT_DESC, false);

        nh.param<double>("odometry_consistency_threshold", odometry_consistency_threshold, 2.0);
        nh.param<double>("pos_covariance_per_meter", pos_covariance_per_meter, 0.01);
        nh.param<double>("yaw_covariance_per_meter", yaw_covariance_per_meter, 0.003);

        nh.param<double>("loop_cov_pos", loop_cov_pos, 0.013);
        nh.param<double>("loop_cov_ang", loop_cov_ang, 2.5e-04);
        nh.param<int>("min_direction_loop", MIN_DIRECTION_LOOP, 3);
        nh.param<std::string>("output_path", OUTPUT_PATH, "");

        //Network config
        nh.param<std::string>("lcm_uri", _lcm_uri, "udpm://224.0.0.251:7667?ttl=1");
        nh.param<bool>("send_all_features", SEND_ALL_FEATURES, false);


        //Read VINS params.
        cv::FileStorage fsSettings;
        fsSettings.open(vins_config_path.c_str(), cv::FileStorage::READ);

        if (camera_configuration == CameraConfig::STEREO_PINHOLE) {
            MAX_DIRS = 1;
        } else if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
            MAX_DIRS = 4;
        } else if (camera_configuration == CameraConfig::PINHOLE_DEPTH) {
            MAX_DIRS = 1;
            fsSettings["depth_topic"] >> DEPTH_TOPIC;
        } else {
            MAX_DIRS = 0;
            ROS_ERROR("[SWARM_LOOP] Camera configuration %d not implement yet.", camera_configuration);
            exit(-1);
        }

        fsSettings["image0_topic"] >> IMAGE0_TOPIC;
        fsSettings["image1_topic"] >> IMAGE1_TOPIC;

        fsSettings["compressed_image0_topic"] >> COMP_IMAGE0_TOPIC;
        fsSettings["compressed_image1_topic"] >> COMP_IMAGE1_TOPIC;

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        left_extrinsic = toROSPose(Swarm::Pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3)));

        fsSettings["body_T_cam1"] >> cv_T;
        cv::cv2eigen(cv_T, T);

        int is_comp_images = 0;
        fsSettings["is_compressed_images"] >> is_comp_images;

        right_extrinsic = toROSPose(Swarm::Pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3)));
    }
}