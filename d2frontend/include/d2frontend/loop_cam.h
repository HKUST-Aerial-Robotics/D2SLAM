#pragma once

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <functional>
#include "d2frontend_params.h"
#include "CNN/onnx_generic.h"
#include "CNN/mobilenetvlad_onnx.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <d2frontend/utils.h>
#include "d2common/d2frontend_types.h"
#include <fstream>
#include <memory>

#include "CNN/superpoint_tensorrt.h"
#include "CNN/superpoint_onnx.h"

//#include <swarm_loop/HFNetSrv.h>

using namespace swarm_msgs;
using namespace Eigen;
using namespace D2Common;

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
    std::string superpoint_model;
    std::string pca_comp;
    std::string pca_mean;
    double superpoint_thres;
    int superpoint_max_num;
    std::string netvlad_model;
    int self_id = 0;
    bool OUTPUT_RAW_SUPERPOINT_DESC;
    bool right_cam_as_main = false;
    double DEPTH_NEAR_THRES;
    double TRIANGLE_THRES;
    int ACCEPT_MIN_3D_PTS;
    double DEPTH_FAR_THRES;
    bool stereo_as_depth_cam = false;
    bool cnn_use_onnx = true;
    bool send_img;
    bool show = false;
    bool cnn_enable_tensorrt = false;
    bool cnn_enable_tensorrt_int8 = false;
    bool cnn_enable_tensorrt_fp16 = true;
    bool enable_undistort_image; //Undistort image before feature detection
    std::string netvlad_int8_calib_table_name;
    std::string superpoint_int8_calib_table_name;

    SuperPointConfig superpoint_config;
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
    std::vector<FisheyeUndist*> undistortors;
    MobileNetVLADONNX * netvlad_onnx = nullptr;
#ifdef USE_CUDA
    std::unique_ptr<SuperPoint> superpoint_ptr = nullptr;
#else
    SuperPointONNX * superpoint_ptr = nullptr;
#endif
public:
    // LoopDetector * loop_detector = nullptr;
    LoopCam(LoopCamConfig config, ros::NodeHandle & nh);
    
    VisualImageDesc extractorImgDescDeepnet(ros::Time stamp, cv::Mat img, int index, int camera_id, bool superpoint_mode=false);
    std::vector<VisualImageDesc> generateStereoImageDescriptor(const StereoFrame & msg, int i, cv::Mat &_show);
    VisualImageDesc generateGrayDepthImageDescriptor(const StereoFrame & msg, int i, cv::Mat &_show);
    VisualImageDesc generateImageDescriptor(const StereoFrame & msg, int i, cv::Mat &_show);
    VisualImageDescArray processStereoframe(const StereoFrame & msg);

    void encodeImage(const cv::Mat & _img, VisualImageDesc & _img_desc);
    
    std::vector<camodocal::CameraPtr> cams;

    CameraConfig getCameraConfiguration() const {
        return camera_configuration;
    }

};
}
