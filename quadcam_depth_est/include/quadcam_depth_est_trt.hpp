#pragma once

#include <swarm_msgs/Pose.h>
#include <opencv2/cudaimgproc.hpp>
#include "virtual_stereo.hpp"
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
#include <image_transport/image_transport.h>
#include "pcl_utils.hpp"
#include <d2common/d2basetypes.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/time_synchronizer.h>
#include <thread>
#include "hitnet.hpp"

namespace D2QuadCamDepthEst{
const int32_t kCamerasNum = 4;
using D2Common::CameraConfig;

cv::Mat quadReadVingette(const std::string & mask_file, double avg_brightness);

class QuadcamDepthEstTrt{
 public:
  QuadcamDepthEstTrt(ros::NodeHandle & nh);
  ~QuadcamDepthEstTrt();
  void startAllService();
  void stopAllService();
 private:
  void quadcamImageCb(const sensor_msgs::ImageConstPtr & images);
  void loadVirtualCameras(YAML::Node & config, std::string configPath);
  
  void rawImageProcessThread();
  void inferrenceThread();
  void publishThread();
  void stoprawImageProcessThread(){
    raw_image_process_thread_running_ = 0;
  };
  void stopinfrenceThread(){
    inference_thread_running_ = 0;
  };
  void stoppublishThread(){
    publish_thread_running_ = 0;
  };

  std::pair<cv::Mat, cv::Mat> intrinsicsFromNode(const YAML::Node & node);

  std::vector<camodocal::CameraPtr> raw_cameras_; //fisheye cameras
  std::vector<D2Common::FisheyeUndist*> undistortors_; //undistortors
  std::vector<VirtualStereo*> virtual_stereos_; //virtual stereo

  std::vector<Swarm::Pose> raw_cam_extrinsics_; //extrinsics of fisheye cameras
  std::vector<Swarm::Pose> virtual_left_extrinsics_; //extrinsics of virtual left cameras

  std::unique_ptr<TensorRTHitnet::HitnetTrt> hitnet_ = nullptr; //hitnet
  ros::NodeHandle nh_;
  image_transport::ImageTransport * image_transport_;
  image_transport::Subscriber image_sub_;

  ros::Publisher pub_pcl_;
  PointCloud * pcl_ = nullptr;
  PointCloudRGB * pcl_color_ = nullptr;
  CameraConfig camera_config_ = D2Common::FOURCORNER_FISHEYE;

  //configurations
  //image size
  int width_= 320;
  int height_= 240;
  int pixel_step_ = 1;
  int image_step_ = 1;
  bool enable_texture_ = false;
  bool show_ = false;
  double min_z_ = 0.1;
  double max_z_ = 10;
  int image_count_ = 0;
  bool cnn_input_rgb_ = false;

  int32_t fps_ = 10;
  std::string onnx_path_;
  std::string trt_engine_path_;
  
  std::thread raw_image_process_thread_;
  std::thread inference_thread_;
  std::thread publish_thread_;
  int32_t raw_image_process_thread_running_ = 1;
  int32_t inference_thread_running_ = 1;
  int32_t publish_thread_running_ = 1;

  //ros rate
  std::unique_ptr<ros::Rate> raw_image_process_rate_ = nullptr;
  std::unique_ptr<ros::Rate> inference_rate_ = nullptr;
  std::unique_ptr<ros::Rate> publish_rate_ = nullptr;

  std::string image_format_ = "raw";
  std::string image_topic_ = "/oak_ffc_4p/assemble_image";
  const std::string kPointCloudTopic_ = "/depth_estimation/pointcloud";

  //buffers to store images double buffe
  cv::Mat raw_image_;
  std_msgs::Header raw_image_header_;
  std::mutex raw_image_mutex_;

  cv::Mat split_raw_images_[kCamerasNum];
  cv::cuda::GpuMat rectified_images_[kCamerasNum][2];//{left,right},{left,right},{left,right},{left,right}
  cv::Mat input_tensors_[kCamerasNum];
  std::mutex input_tensors_mutex_;

  cv::Mat output_tensors_[kCamerasNum];
  std::mutex output_tensors_mutex_;

  cv::Mat recity_images_for_show_and_texture_[kCamerasNum][2];
  cv::Mat publish_disparity_[kCamerasNum];
  cv::Mat photometric_inv_vingette_[kCamerasNum];

};

}