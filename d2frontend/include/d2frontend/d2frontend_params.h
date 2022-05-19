#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>

#define ACCEPT_LOOP_YAW (30) //ACCEPT MAX Yaw 
#define MAX_LOOP_DIS 5.0 //ACCEPT MAX DISTANCE, 2.0 for indoor flying

#define MAX_LOOP_DIS_LEVEL2 3.0 //ACCEPT MAX DISTANCE, 2.0 for indoor flying

#define DEG2RAD (0.01745277777777778)

#define ACCEPT_LOOP_YAW_RAD ACCEPT_LOOP_YAW*DEG2RAD

#define USE_DEEPNET

#define DEEP_DESC_SIZE 1024
#define NETVLAD_DESC_SIZE 4096

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

namespace D2FrontEnd {
enum CameraConfig{
    STEREO_PINHOLE = 0,
    STEREO_FISHEYE = 1,
    PINHOLE_DEPTH = 2,
    FOURCORNER_FISHEYE = 3
};

struct LoopCamConfig;
struct LoopDetectorConfig;
struct D2FTConfig;

struct D2FrontendParams {
    int JPG_QUALITY;
    double ACCEPT_NONKEYFRAME_WAITSEC;
    bool USE_DEPTH;
    bool SEND_ALL_FEATURES;
    std::string OUTPUT_PATH;
    int width;
    int height;
    double max_freq = 1.0;
    double recv_msg_duration = 0.5;
    double feature_min_dist = 20;
    int total_feature_num = 150;
    double vlad_threshold = 0.3;

    double min_movement_keyframe = 0.3;
    int self_id = 0;
    std::string vins_config_path;
    std::string _lcm_uri = "0.0.0.0";
    CameraConfig camera_configuration;

    D2FrontendParams(ros::NodeHandle &);

    //Debug params
    bool send_img;
    bool enable_pub_remote_frame;
    bool enable_pub_local_frame;
    bool enable_sub_remote_frame;
    bool send_whole_img_desc;
    bool enable_perf_output = false;
    bool debug_image = false;
    bool debug_plot_superpoint_features = false;
    bool enable_loop = true;
    bool enable_network = true;

    bool is_comp_images;
    std::vector<std::string> image_topics, comp_image_topics, depth_topics;

    //Extrinsics
    std::vector<Swarm::Pose> extrinsics;

    LoopCamConfig * loopcamconfig;
    LoopDetectorConfig * loopdetectorconfig;
    D2FTConfig * ftconfig;

};
extern D2FrontendParams * params;

}