#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <swarm_msgs/Pose.h>
#include <d2common/d2basetypes.h>
#include <yaml-cpp/yaml.h>


#define ACCEPT_LOOP_YAW (30) //ACCEPT MAX Yaw 

#define MAX_LOOP_DIS_LEVEL2 3.0 //ACCEPT MAX DISTANCE, 2.0 for indoor flying

#define DEG2RAD (0.01745277777777778)

#define USE_DEEPNET

#define SEARCH_NEAREST_NUM 5
// #define ACCEPT_NONKEYFRAME_WAITSEC 5.0
#define INIT_ACCEPT_NONKEYFRAME_WAITSEC 1.0

#define VISUALIZE_SCALE 2 //Scale for visuallize

#define CROP_WIDTH_THRES 0.05 //If movement bigger than this, crop some matches down


#define ACCEPT_SP_MATCH_DISTANCE 0.7

namespace camodocal {
class Camera;
typedef boost::shared_ptr< Camera > CameraPtr;
}

namespace D2Common {
class FisheyeUndist;
}

namespace D2FrontEnd {
using D2Common::CameraConfig;
using D2Common::ESTIMATION_MODE;

enum TrackLRType {
    WHOLE_IMG_MATCH = 0,
    LEFT_RIGHT_IMG_MATCH,
    RIGHT_LEFT_IMG_MATCH
};

struct LoopCamConfig;
struct LoopDetectorConfig;
struct D2FTConfig;

struct D2FrontendParams {
    int JPG_QUALITY;
    double ACCEPT_NONKEYFRAME_WAITSEC;
    bool USE_DEPTH;
    std::string OUTPUT_PATH;
    int width;
    int height;
    int image_queue_size; //this size is critical for the realtime performance
    double recv_msg_duration = 0.5;
    double feature_min_dist = 20;
    int total_feature_num = 150;
    double track_remote_netvlad_thres = 0.3;
    size_t superpoint_dims = 256;
    size_t netvlad_dims = 4096;
    bool enable_pca_superpoint = false;
    bool enable_pca_netvlad = false;

    std::string pca_netvlad = "";

    double min_movement_keyframe = 0.3;
    int self_id = 0;
    std::string vins_config_path;
    std::string _lcm_uri = "0.0.0.0";
    CameraConfig camera_configuration;
    int min_receive_images = 2;

    D2Common::PGO_MODE pgo_mode;
    ESTIMATION_MODE estimation_mode;

    //Debug params
    bool send_img;
    bool enable_pub_remote_frame;
    bool enable_pub_local_frame;
    bool enable_sub_remote_frame;
    bool send_whole_img_desc;
    bool enable_perf_output = false;
    bool show = false;
    bool debug_plot_superpoint_features = false;
    bool enable_loop = true;
    bool enable_network = true;
    bool verbose = false;
    bool print_network_status = false;
    bool lazy_broadcast_keyframe = true;

    bool is_comp_images;
    std::vector<std::string> image_topics, depth_topics;

    //Extrinsics and camera configs
    double undistort_fov = 200;
    int width_undistort = 800;
    int height_undistort = 400;
    bool enable_undistort_image; //Undistort image before feature detection
    double focal_length = 460.0;
    std::vector<Swarm::Pose> extrinsics;
    std::vector<cv::Mat> cam_Ks;
    std::vector<cv::Mat> cam_Ds;
    std::vector<double> cam_xis;
    std::vector<std::string> camera_config_paths;
    std::vector<camodocal::CameraPtr> camera_ptrs;
    std::vector<camodocal::CameraPtr> raw_camera_ptrs;
    std::vector<D2Common::FisheyeUndist*> undistortors;
    std::vector<int> camera_seq;

    bool show_raw_image = false;

    //Configs of submodules
    LoopCamConfig * loopcamconfig;
    LoopDetectorConfig * loopdetectorconfig;
    D2FTConfig * ftconfig;

    D2FrontendParams(ros::NodeHandle &);
    D2FrontendParams() {}
    void readCameraCalibrationfromFile(const std::string & path, int32_t extrinsic_parameter_type = 1);
    void generateCameraModels(cv::FileStorage & fsSettings, std::string config_path);
    void readCameraConfigs(cv::FileStorage & fsSettings, std::string config_path);
    static std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(
    const std::string& camera_name, const YAML::Node& config, int32_t extrinsic_parameter_type = 1);


};
extern D2FrontendParams * params;

}