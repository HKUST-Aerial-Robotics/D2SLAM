#include "quadcam_depth_est_trt.hpp"
#include <d2common/fisheye_undistort.h>
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <d2frontend/d2frontend_params.h>
#include <NvInferRuntime.h>
#include <spdlog/spdlog.h>

#include "pcl_utils.hpp"


// namespace D2FrontEnd {
//     std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config, int32_t extrinsic_parameter_type = 1);
// };

namespace D2QuadCamDepthEst
{

cv::Mat quadReadVingette(const std::string & mask_file, double avg_brightness) {
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

QuadcamDepthEstTrt::QuadcamDepthEstTrt(ros::NodeHandle & nh):nh_(nh){
  std::string config_file_path;
  nh.getParam("depth_config",config_file_path);
  nh.getParam("show",show_);
  printf("[QuadCamDepthEstTrt]:read config from:%s show: %d\n",config_file_path.c_str(),show_);
  YAML::Node config = YAML::LoadFile(config_file_path); 
  std::string config_dir = config_file_path.substr(0,config_file_path.find_last_of("/"));
  this->enable_texture_ = config["enable_texture"].as<bool>();
  this->pixel_step_ = config["pixel_step"].as<int>();
  this->image_step_ = config["image_step"].as<int>();
  this->min_z_ = config["min_z"].as<double>();
  this->max_z_ = config["max_z"].as<double>();
  this->width_ = config["width"].as<int>();
  this->height_ = config["height"].as<int>();
  this->fps_ = config["fps"].as<int>();
  this->raw_image_process_rate_ = std::make_unique<ros::Rate>(ros::Rate(this->fps_));
  this->inference_rate_ = std::make_unique<ros::Rate>(ros::Rate(this->fps_));
  this->publish_rate_ = std::make_unique<ros::Rate>(ros::Rate(this->fps_));
  this->cnn_input_rgb_ = config["cnn_input_rgb"].as<bool>();
  
  this->loadVirtualCameras(config,config_dir);
  if(config["image_topic"].IsDefined()){
    this->image_topic_ = config["image_topic"].as<std::string>();
  }
  if(config["image_format"].IsDefined()){
    this->image_format_ = config["image_format"].as<std::string>();
  }
  //hitnet
  for(int i = 0 ; i<kCamerasNum; i++){
    this->output_tensors_[i] = cv::Mat(this->height_,this->width_,CV_32F);
  }
  this->onnx_path_ = config["onnx_path"].as<std::string>();
  this->trt_engine_path_ = config["trt_engine_path"].as<std::string>();

  this->hitnet_ = std::make_unique<TensorRTHitnet::HitnetTrt>(true);
  this->hitnet_->init(onnx_path_,trt_engine_path_, 4);

  //subscribe
  image_transport::TransportHints hints(this->image_format_, ros::TransportHints().tcpNoDelay(true));
  image_transport_ = new image_transport::ImageTransport(nh_);
  image_sub_ = image_transport_->subscribe(this->image_topic_, 1, &QuadcamDepthEstTrt::quadcamImageCb, this, hints);
  if (enable_texture_){
    pcl_color_ = new PointCloudRGB();
    pcl_color_->points.reserve(virtual_stereos_.size() * width_ * height_);
  } else {
    pcl_ = new PointCloud();
    pcl_->points.reserve(virtual_stereos_.size() * width_ * height_);
  }

  //publisher
  this->pub_pcl_ = nh_.advertise<sensor_msgs::PointCloud2>(kPointCloudTopic_, 1);
  printf("QuadcamDepthEtsTrt constructed\n");
};

QuadcamDepthEstTrt::~QuadcamDepthEstTrt(){
  if(pcl_ != nullptr){
    delete pcl_;
    pcl_ = nullptr;
  }
  if(pcl_color_ != nullptr){
    delete pcl_color_;
    pcl_color_ = nullptr;
  }
  if (this->hitnet_ != nullptr){
    this->hitnet_ = nullptr;
  }
};

void QuadcamDepthEstTrt::loadVirtualCameras(YAML::Node & config, std::string configPath){
  float avg_brightness = config["avg_brightness"].as<float>();
  std::string photometric_calib_path = config["photometric_calib_path"].as<std::string>();
  //Read photometric calibration masks
  if (access(photometric_calib_path.c_str(),F_OK) == 0){
    printf("[QuadcamDepthEstTrt]: loadVirtualCameras from %s\n",photometric_calib_path.c_str());
    for(int i=0 ; i < kCamerasNum ; i++){
      std::string mask_file = photometric_calib_path + "/" + std::string("cam_") + std::to_string(i) + std::string("_vig_mask.png");//search image "cam_i_vig_mask.png"
      if(access(mask_file.c_str(),F_OK) == 0){
        photometric_inv_vingette_[i] = quadReadVingette(mask_file, avg_brightness);
        printf("[QuadcamDepthEstTrt]: read vignette mask from %s\n",mask_file.c_str());
      } else {
        photometric_inv_vingette_[i] = cv::Mat (1280, 720, CV_8UC3, cv::Scalar(255, 255, 255));
      }
    }
  } else {
    for(int i=0 ; i < kCamerasNum ; i++){
      photometric_inv_vingette_[i] = cv::Mat (1280, 720, CV_8UC3, cv::Scalar(255, 255, 255));
    }
  }
  //Read fisheye cameras intrinsic and extrinsic parameters
  std::string cam_calib_file_path = config["cam_calib_file_path"].as<std::string>();
  printf("[QuadcamDepthEstTrt]: load camera calibration from %s\n",cam_calib_file_path.c_str());
  if(access(cam_calib_file_path.c_str(),R_OK) == 0){
    YAML::Node fisheye_configs = YAML::LoadFile(cam_calib_file_path);
    int32_t photometric_inv_idx = 0;
    for (const auto & cam_para : fisheye_configs){
      std::string camera_name = cam_para.first.as<std::string>();
      printf("[QuadcamDepthEstTrt] Load camera %s\n", camera_name.c_str());
      //fisheye camera parameters
      const YAML::Node & camera_parameters = cam_para.second;
      auto cam_model = D2FrontEnd::D2FrontendParams::readCameraConfig(camera_name,camera_parameters);
      this->raw_cameras_.push_back(cam_model.first); 
      //load distotors and photometric calibration
      double fov = config["fov"].as<double>();
      if(photometric_inv_idx >=4 || photometric_inv_idx < 0){
          photometric_inv_idx = 0;
      }
      printf("[Debug ]undistortor matrix init with size width:%d height:%d\n",this->width_,this->height_);
      this->undistortors_.push_back(new D2Common::FisheyeUndist(cam_model.first, 0, fov,
          D2Common::FisheyeUndist::UndistortPinhole2, this->width_, this->height_, photometric_inv_vingette_[photometric_inv_idx]));
      printf("[Debug] undistorter width and height:%d %d\n",this->width_,this->height_);
      photometric_inv_idx++;
      //set extrinsic paratmers
      this->raw_cam_extrinsics_.push_back(cam_model.second);
    }

    //Create virtual stereo
    for(const auto & vstereos:config["stereos"]){
      auto stereo_node =  vstereos.second;
      std::string stereo_name = vstereos.first.as<std::string>();
      int cam_idx_l = stereo_node["cam_idx_l"].as<int>();
      int cam_idx_r = stereo_node["cam_idx_r"].as<int>();
      int idx_l = stereo_node["idx_l"].as<int>();
      int idx_r = stereo_node["idx_r"].as<int>();
      std::string stereo_calib_file = stereo_node["stereo_config"].as<std::string>();
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
          undistortors_[cam_idx_l], undistortors_[cam_idx_r], idx_l, idx_r);
      auto att = undistortors_[cam_idx_l]->t[idx_l];
      stereo->extrinsic = raw_cam_extrinsics_[cam_idx_l] * Swarm::Pose(att, Vector3d(0, 0, 0));
      stereo->enable_texture = enable_texture_;
      stereo->initRecitfy(baseline, KD0.first, KD0.second, KD1.first, KD1.second);
      virtual_stereos_.emplace_back(stereo);
    }
    printf("[QuadCamDepthEst] Init virtual cameras successfully\n");
  } else {
    printf("QuadcamDepthEstTrt][Failed]: read camera calibration from %s\n",cam_calib_file_path.c_str());
    return ;
  }
  return ;
}

std::pair<cv::Mat, cv::Mat> QuadcamDepthEstTrt::intrinsicsFromNode(const YAML::Node & node) {
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

void QuadcamDepthEstTrt::startAllService(){
  this->raw_image_process_thread_ = std::thread(&QuadcamDepthEstTrt::rawImageProcessThread,this);
  this->inference_thread_ = std::thread(&QuadcamDepthEstTrt::inferrenceThread,this);
  this->publish_thread_ = std::thread(&QuadcamDepthEstTrt::publishThread,this);
  printf("[QuadcamDepthEstTrt]: start all service\n");
}

void QuadcamDepthEstTrt::stopAllService(){
  if(this->inference_thread_.joinable()){
    this->stopinfrenceThread();
    this->inference_thread_.join();
  }
  if(this->publish_thread_.joinable()){
    this->stoppublishThread(); 
    this->publish_thread_.join();
  }
  if(this->raw_image_process_thread_.joinable()){
    this->stoprawImageProcessThread();
    this->raw_image_process_thread_.join();
  }
}

void QuadcamDepthEstTrt::quadcamImageCb(const sensor_msgs::ImageConstPtr & images){
  if (!raw_image_mutex_.try_lock()){
    return;
  } else {
    raw_image_ = cv_bridge::toCvCopy(images, sensor_msgs::image_encodings::BGR8)->image;
    this->raw_image_header_ = images->header;
    raw_image_mutex_.unlock();
  }
  return;
}

//TODO:kCamearsNum = size of vitual_stereos_
void QuadcamDepthEstTrt::rawImageProcessThread(){
  while(raw_image_process_thread_running_){
    static cv::Mat raw_image;
    /* Because raw_image_ always get new memory addr, 
      so here we handle the memory and release raw_image_ for cb */
    if (raw_image_mutex_.try_lock()){
      if (raw_image_.empty()){
        this->raw_image_mutex_.unlock();
        this->raw_image_process_rate_->sleep();
        continue;
      } else {
        raw_image = raw_image_;
        this->raw_image_mutex_.unlock();
      }
    } else {
      this->raw_image_process_rate_->sleep();
      continue;
    }

    for(int32_t i = 0; i< kCamerasNum; i++){
      cv::Mat splited_image = raw_image(cv::Rect(i * raw_image_.cols /kCamerasNum, 0, 
        raw_image.cols /kCamerasNum, raw_image.rows));
      if(!this->cnn_input_rgb_){
        
        if (splited_image.empty()){
          printf("[QuadcamDepthEstTrt]: splited image is empty\n");
          this->raw_image_process_rate_->sleep();
          continue;
        }
        
        cv::cvtColor(splited_image,split_raw_images_[i],cv::COLOR_BGR2GRAY);//TODO: Bug openCV segement fault
        #ifdef DEBUG
        printf("[QuadcamDepthEstTrt]: split raw image to gray\n");
        char window_name[20];
        sprintf(window_name,"raw_image_%d",i);
        cv::imshow(window_name,split_raw_images_[i]);
        cv::waitKey(1);
        #endif
      } else {
        split_raw_images_[i] = splited_image;
      }
    }
    #ifdef DEBUG
    // printf("[QuadcamDepthEstTrt]: split raw image\n");
    cv::imshow("raw_image",raw_image_);
    cv::waitKey(1);
    #endif

    //get rectify images
    for(auto && stereo: this->virtual_stereos_){
      stereo->rectifyImage(split_raw_images_[stereo->cam_idx_a],split_raw_images_[stereo->cam_idx_b],
        rectified_images_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id],
        rectified_images_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id]);
    }

    #ifdef DEBUG
    //show all pairs of rectified images
    for(auto && stereo: this->virtual_stereos_){
      char window_name[20];
      cv::Mat show_image;
      cv::Mat left(rectified_images_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id]);
      cv::Mat right(rectified_images_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id]);
      cv::hconcat(left,right,show_image);
      sprintf(window_name,"rectified_image_%d_%d",stereo->cam_idx_a,stereo->cam_idx_b);
      cv::imshow(window_name,show_image);
      cv::waitKey(1);
    }
    #endif

    //construct input images for hitnet inferrence and  TODO: can gpu mat be used directly?
    cv::Mat temp_left , temp_right, input_image[4];

    for (auto && stereo : this->virtual_stereos_){
      temp_left = cv::Mat(rectified_images_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id]);
      temp_right = cv::Mat(rectified_images_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id]);
      recity_images_for_show_and_texture_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id] = temp_left;
      recity_images_for_show_and_texture_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id] = temp_right;
      // redundant undistort image is already in size
      // cv::resize(temp_left,temp_left,cv::Size(this->width_,this->height_));
      // cv::resize(temp_right,temp_right,cv::Size(this->width_,this->height_));
      cv::vconcat(temp_left,temp_right,input_image[stereo->stereo_id]);
    }

    //to reduce the time of mutex lock
    if (!input_tensors_mutex_.try_lock()){
      this->raw_image_process_rate_->sleep();
      continue;
    } else {
      for (auto && stereo : this->virtual_stereos_){
        input_image[stereo->stereo_id].convertTo(input_tensors_[stereo->stereo_id],CV_32FC1,1.0/255.0);
      }
      input_tensors_mutex_.unlock();
    }
    this->raw_image_process_rate_->sleep();
  }
  return ;
}

void QuadcamDepthEstTrt::inferrenceThread(){
  static cv::Mat input_tensors[4];
  while(inference_thread_running_){
    if(input_tensors_mutex_.try_lock()){
      //if input_tensors_ is empty, wait for next loop
      if (this->input_tensors_[0].empty()){
        this->input_tensors_mutex_.unlock();
        this->inference_rate_->sleep();
        continue;
      }

      for (auto stereo : this->virtual_stereos_){
        input_tensors[stereo->stereo_id] = input_tensors_[stereo->stereo_id];
      }
      input_tensors_mutex_.unlock();
    } else {
      this->inference_rate_->sleep();
      continue;
    }
    this->hitnet_->doInference(input_tensors);

    if (output_tensors_mutex_.try_lock()){
      this->hitnet_->getOutput(output_tensors_);
      output_tensors_mutex_.unlock();
    } else {
      this->inference_rate_->sleep();
      continue;
    }
    this->inference_rate_->sleep();
  }
  return ;
}

void QuadcamDepthEstTrt::publishThread(){
  //TODO: publish pointcloud and do visualization
  while(publish_thread_running_){
    //if output_tensors_ is empty, wait for next loop
    if (this->output_tensors_[0].empty()){
      this->publish_rate_->sleep();
      continue;
    }

    //copy data to local
    if (output_tensors_mutex_.try_lock()){
      for (auto stereo : this->virtual_stereos_){
        publish_disparity_[stereo->stereo_id] = output_tensors_[stereo->stereo_id];
      }
      output_tensors_mutex_.unlock();
    } else {
      this->publish_rate_->sleep();
      continue;
    }
    //debug show disparity
    if(show_){
      if (recity_images_for_show_and_texture_[0][0].empty()){
        this->publish_rate_->sleep();
        continue;
      }
      for (auto stereo : this->virtual_stereos_){
        stereo->showDispartiy(publish_disparity_[stereo->stereo_id], 
          recity_images_for_show_and_texture_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id],
          recity_images_for_show_and_texture_[stereo->cam_idx_b][stereo->cam_idx_b_left_half_id]);
      }
    }
    //prepeare pcl
    if (pcl_ == nullptr){
      printf("[QuadcamDepthEstTrt]: pcl is nullptr\n");
      this->publish_rate_->sleep();
      continue;
    }
    pcl_conversions::toPCL(raw_image_header_.stamp, pcl_->header.stamp);
    pcl_->header.frame_id = "imu";
    pcl_->points.clear();
    //TODO: if enable texture 
    for (auto stereo : this->virtual_stereos_){
      cv::Mat points;
      cv::reprojectImageTo3D(publish_disparity_[stereo->stereo_id], 
        points, stereo->getStereoPose(), 3);
      addPointsToPCL(points,recity_images_for_show_and_texture_[stereo->cam_idx_a][stereo->cam_idx_a_right_half_id], 
        stereo->extrinsic, *this->pcl_, this->pixel_step_, this->min_z_, this->max_z_);  
    }
    pub_pcl_.publish(*pcl_);
    this->publish_rate_->sleep();
  }
  return ;
}

} // namespace D2QuadCamDepthEst
