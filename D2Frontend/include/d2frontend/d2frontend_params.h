#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <ros/ros.h>

#define LOOP_BOW_THRES 0.015
// #define MATCH_INDEX_DIST 1
#define FAST_THRES (20.0f)


#define ACCEPT_LOOP_YAW (30) //ACCEPT MAX Yaw 
#define MAX_LOOP_DIS 5.0 //ACCEPT MAX DISTANCE, 2.0 for indoor flying


#define MAX_LOOP_DIS_LEVEL2 3.0 //ACCEPT MAX DISTANCE, 2.0 for indoor flying

#define DEG2RAD (0.01745277777777778)

#define ACCEPT_LOOP_YAW_RAD ACCEPT_LOOP_YAW*DEG2RAD

#define USE_DEEPNET

#define DEEP_DESC_SIZE 1024

#define SEARCH_NEAREST_NUM 5
// #define ACCEPT_NONKEYFRAME_WAITSEC 5.0
#define INIT_ACCEPT_NONKEYFRAME_WAITSEC 1.0

#define VISUALIZE_SCALE 2 //Scale for visuallize

#define CROP_WIDTH_THRES 0.05 //If movement bigger than this, crop some matches down

#define OUTLIER_XY_PRECENT_0 0.03 // This is given up match dx dy 
#define OUTLIER_XY_PRECENT_20 0.03 // This is given up match dx dy 
#define OUTLIER_XY_PRECENT_30 0.03 // This is given up match dx dy 
#define OUTLIER_XY_PRECENT_40 0.03 // This is given up match dx dy 

#define RPERR_THRES 10*DEG2RAD


#define FEATURE_DESC_SIZE 64


#define ACCEPT_SP_MATCH_DISTANCE 0.7

namespace D2Frontend {
enum CameraConfig{
    STEREO_PINHOLE = 0,
    STEREO_FISHEYE = 1,
    PINHOLE_DEPTH = 2,
    FOURCORNER_FISHEYE = 3
};


struct D2FrontendParams {
    int JPG_QUALITY;
    int INIT_MODE_MIN_LOOP_NUM; //Init mode we accepte this inlier number
    int MIN_LOOP_NUM;
    double ACCEPT_NONKEYFRAME_WAITSEC;
    double INNER_PRODUCT_THRES;
    double INIT_MODE_PRODUCT_THRES;//INIT mode we can accept this inner product as similar
    int MATCH_INDEX_DIST;
    int ACCEPT_MIN_3D_PTS;
    double DEPTH_NEAR_THRES;
    double DEPTH_FAR_THRES;
    int MAX_DIRS;
    int MIN_DIRECTION_LOOP;
    int MIN_MATCH_PRE_DIR;
    double TRIANGLE_THRES;
    double loop_cov_pos;
    double loop_cov_ang;
    double DETECTOR_MATCH_THRES;
    double odometry_consistency_threshold;
    bool USE_DEPTH;
    bool IS_PC_REPLAY;
    bool SEND_ALL_FEATURES;
    bool LOWER_CAM_AS_MAIN;
    bool OUTPUT_RAW_SUPERPOINT_DESC;
    double pos_covariance_per_meter;
    double yaw_covariance_per_meter;
    std::string OUTPUT_PATH;
    bool DEBUG_NO_REJECT;
    int width;
    int height;
    int inter_drone_init_frames;
    bool is_4dof;
    double max_freq = 1.0;
    double recv_msg_duration = 0.5;

    bool debug_image = false;
    double min_movement_keyframe = 0.3;
    int self_id = 0;
    bool received_image = false;
    std::string camera_config_path = "";
    std::string superpoint_model_path = "";
    std::string netvlad_model_path = "";
    std::string vins_config_path;
    std::string _lcm_uri = "0.0.0.0";

    D2FrontendParams(ros::NodeHandle &);

    LoopCamConfig loopcamconfig;
};

extern D2FrontendParams * params;
}