#pragma once

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <functional>
#include "d2frontend_params.h"
#include "superpoint_tensorrt.h"
#include "mobilenetvlad_tensorrt.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <d2frontend/utils.h>
#include "d2frontend/d2frontend_types.h"

//#include <swarm_loop/HFNetSrv.h>

using namespace swarm_msgs;
using namespace Eigen;
namespace camodocal {
class Camera;
typedef boost::shared_ptr< Camera > CameraPtr;
}

namespace D2FrontEnd {
void matchLocalFeatures(std::vector<cv::Point2f> & pts_up, std::vector<cv::Point2f> & pts_down, 
    std::vector<float> & _desc_up, std::vector<float> & _desc_down, 
    std::vector<int> & ids_up, std::vector<int> & ids_down);

struct LoopCamConfig
{
    /* data */
    CameraConfig camera_configuration;
    std::string camera_config_path;
    std::string superpoint_model;
    std::string pca_comp;
    std::string pca_mean;
    double superpoint_thres;
    int superpoint_max_num;
    std::string netvlad_model;
    int width;
    int height; 
    int self_id = 0;
    bool OUTPUT_RAW_SUPERPOINT_DESC;
    bool right_cam_as_main = false;
    double DEPTH_NEAR_THRES;
    double TRIANGLE_THRES;
    int ACCEPT_MIN_3D_PTS;
    double DEPTH_FAR_THRES;
    bool stereo_as_depth_cam = false;
};

class LoopCam {
    LoopCamConfig _config;
    int cam_count = 0;
    int loop_duration = 10;
    int self_id = 0;
    int kf_count = 0;
    ros::ServiceClient hfnet_client;
    ros::ServiceClient superpoint_client;
    CameraConfig camera_configuration;
    std::fstream fsp;
#ifdef USE_TENSORRT
    Swarm::SuperPointTensorRT superpoint_net;
    Swarm::MobileNetVLADTensorRT netvlad_net;
#endif

    bool send_img;

public:

    bool show = false;

    // LoopDetector * loop_detector = nullptr;
    LoopCam(LoopCamConfig config, ros::NodeHandle & nh);
    
    VisualImageDesc extractorImgDescDeepnet(ros::Time stamp, cv::Mat img, int index, int camera_id, bool superpoint_mode=false);
    std::vector<VisualImageDesc> generateStereoImageDescriptor(const StereoFrame & msg, cv::Mat & img, int i, cv::Mat &_show);
    VisualImageDesc generateGrayDepthImageDescriptor(const StereoFrame & msg, cv::Mat & img, int i, cv::Mat &_show);
    VisualImageDescArray processStereoframe(const StereoFrame & msg, std::vector<cv::Mat> & imgs);

    void encodeImage(const cv::Mat & _img, VisualImageDesc & _img_desc);
    
    camodocal::CameraPtr cam;
    cv::Mat cameraMatrix;

    CameraConfig getCameraConfiguration() const {
        return camera_configuration;
    }

};
}
