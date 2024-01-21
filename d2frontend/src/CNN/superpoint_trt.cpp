#include "d2frontend/CNN/superpoint_trt.h"
#include "ATen/Parallel.h"
#include "d2frontend/CNN/superpoint_common.h"
#include "d2common/d2frontend_types.h"
#include "d2common/utils.hpp"

namespace TensorRTSupperPoint{

SuperPointTrt::SuperPointTrt(int32_t stream_number,int32_t width, 
  int32_t height, int32_t min_dist, double thres, int32_t max_num, std::string & engine_path,
  std::string& pca_comp_path, std::string& pca_mean_path, bool show_info): stream_number_(stream_number),
  width_(width), height_(height), nms_dist_(min_dist), thres_(thres), max_num_(max_num), engine_path_(engine_path),
  pca_comp_path_(pca_comp_path),pca_mean_path_(pca_mean_path), show_info_(show_info){
  if (stream_number_ <= 0){
    stream_number = 1;
  }
  if (width_ <= 0){
    width_ = 300;
  }
  if (height_ <= 0){
    height_ = 150;
  }
  return;
}

//create engine
int32_t SuperPointTrt::init(){
  nv_runtime_ptr_ = nvinfer1::createInferRuntime(this->logger_);
  if(nv_runtime_ptr_ == nullptr){
    std::cout << "createInferRuntime failed" << std::endl;
    return -1;
  }
  if(this->engine_path_.empty()){
    std::cout << "engine path is empty" << std::endl;
    return -2;
  }
  std::ifstream engine_file(this->engine_path_, std::ios::binary);
  if(!engine_file){
    std::cout << "open engine file failed" << std::endl;
    return -3;
  }
  std::stringstream engine_buffer;
  engine_buffer << engine_file.rdbuf();
  std::string plan = engine_buffer.str();
  printf("engine plan size %d\n", plan.size());
  auto engine = nv_runtime_ptr_->deserializeCudaEngine(plan.data(), plan.size(), nullptr);
  if (engine == nullptr) {
    std::cout << "deserializeCudaEngine failed" << std::endl;
    return -4;
  }

  //create engine
  std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr(engine, InferDeleter());
  nv_engine_ptr_ = engine_ptr;
  if(nv_engine_ptr_ == nullptr){
    std::cout << "create engine failed" << std::endl;
    return -5;
  }

  //create executors
  for(int32_t i = 0; i < stream_number_; i++){
    SuperPointExcutor executor;
    int32_t ret = executor.init(nv_engine_ptr_, {"image"}, {"semi", "desc"});
    if(ret != 0){
      std::cout << "init executor failed" << std::endl;
      return -6;
    }
    executors_.push_back(executor);
  }

  //init output ptr
  for(int32_t i = 0; i < stream_number_; i++){
    float* semi_out_put_ptr = nullptr;
    float* desc_out_put_ptr = nullptr;
    int32_t semi_output_size = executors_[i].getTensorSize(this->output_tensor_names_[0]);
    semi_out_put_ptr = (float*)malloc(semi_output_size);
    int32_t desc_output_size = executors_[i].getTensorSize(this->output_tensor_names_[1]);
    desc_out_put_ptr = (float*)malloc(desc_output_size);
    if(semi_out_put_ptr == nullptr || desc_out_put_ptr == nullptr){
      std::cout << "malloc output ptr failed" << std::endl;
      return -7;
    }
    semi_out_put_ptr_.push_back(semi_out_put_ptr);
    desc_out_put_ptr_.push_back(desc_out_put_ptr);
  }
  printf("TRT Supper point init success\n");
  return 0;
}

int32_t SuperPointTrt::doInference(std::vector<cv::Mat>& inputs){
  //
  if(inputs.size() != stream_number_){
    std::cout << "input size not match" << std::endl;
    return -1;
  }

  for(int32_t i = 0; i < stream_number_; i++){
    cv::Mat input;
    if(inputs[i].rows != height_ || inputs[i].cols != width_){
      cv::resize(inputs[i], inputs[i], cv::Size(width_, height_));
    }
    inputs[i].convertTo(input, CV_32F, 1/255.0);
    int32_t ret = executors_[i].setInputImages(input);
    if(ret != 0){
      std::cout << "set input images failed" << std::endl;
      return -2;
    }
  }

  for(int32_t i = 0; i < stream_number_; i++){
    int32_t ret = executors_[i].doInference();
    if(ret != 0){
      std::cout << "do inference failed" << std::endl;
      return -3;
    }
  }
  
  for(int32_t i = 0; i < stream_number_; i++){
    int32_t ret = executors_[i].copyBack();
    if(ret != 0){
      std::cout << "copy back failed" << std::endl;
      return -4;
    }
  }

  for(int32_t i = 0; i < stream_number_; i++){
    int32_t ret = executors_[i].synchronize();
    if(ret != 0){
      std::cout << "synchronize failed" << std::endl;
      return -5;
    }
  }
  return 0;
}

//TODO:21.Nov Output API alignment
int32_t SuperPointTrt::getOuput(const int drone_id, const D2Common::StereoFrame & msg,D2Common::VisualImageDescArray & viokf){
  D2Common::Utility::TicToc tic;
  for(int i = 0; i < executors_.size(); i++){
    if(semi_out_put_ptr_[i] == nullptr || desc_out_put_ptr_[i] == nullptr){
      std::cout << "output ptr is null" << std::endl;
      return -1;
    }
    executors_[i].getOutput(semi_out_put_ptr_[i], desc_out_put_ptr_[i]);
  }
  double cost = tic.toc();
  // printf("[superpoint trt] get output cost %f\n", cost);
  //process output
  // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kGPU);
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  double stamp = msg.stamp.toSec();
  for(int i = 0; i < executors_.size(); i++){
    auto mProb = at::from_blob(semi_out_put_ptr_[i], {1, 1, height_, width_}, options);
    auto mDesc = at::from_blob(desc_out_put_ptr_[i], {1, kSpuerpointDesRawLen, height_/8, width_/8}, options);
    cv::Mat prob(height_, width_, CV_32FC1, semi_out_put_ptr_[i]);
    //process prob and desc
    std::vector<cv::Point2f> keypoints;
    D2Common::VisualImageDesc vframe;
    vframe.camera_id = msg.left_camera_ids[i];
    vframe.camera_index = msg.left_camera_indices[i];
    vframe.stamp = stamp;
    vframe.drone_id = drone_id;

    tic.tic();
    D2FrontEnd::getKeyPoints(prob,thres_,nms_dist_,keypoints, vframe.landmark_scores, width_,height_,max_num_, vframe.camera_id);
    cost = tic.toc();
    // printf("[superpoint trt] get keypoints cost %f\n", cost);
    
    tic.tic();
    D2FrontEnd::computeDescriptors(mProb, mDesc, keypoints,  vframe.landmark_descriptor, width_, height_, pca_comp_T, pca_mean);
    cost = tic.toc();
    // printf("[superpoint trt] compute descriptors cost %f\n", cost);

    vframe.key_points = keypoints;
    viokf.images[i] = vframe; //TODO: A copy here may be too much
  }
  return 0;
}

int32_t SuperPointExcutor::init(std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr, 
    std::vector<std::string> input_tensor_names,
    std::vector<std::string> output_tensor_names){
  this->engine_ptr_ = engine_ptr;
  if(this->engine_ptr_ == nullptr){
    std::cout << "engine ptr is null" << std::endl;
    return -1;
  }
  //create context
  this->nv_context_ptr_ = this->engine_ptr_->createExecutionContext();
  if(this->nv_context_ptr_ == nullptr){
    std::cout << "create context failed" << std::endl;
    return -2;
  }
  cudaStreamCreate(&stream_);
  this->buffer_manager_ptr_  = std::make_shared<CudaMemoryManager::BufferManager>(this->engine_ptr_,
    1, this->nv_context_ptr_);
  if(this->buffer_manager_ptr_ == nullptr){
    std::cout << "create buffer manager failed" << std::endl;
    return -3;
  }
  this->input_tensor_names_ = input_tensor_names;
  this->output_tensor_names_ = output_tensor_names;
  auto ret = initTensorMap(this->input_tensor_names_, this->input_tensor_map_);
  if(ret != 0){
    std::cout << "init input tensor map failed" << std::endl;
    return -4;
  }
  ret = initTensorMap(this->output_tensor_names_, this->output_tensor_map_);
  if(ret != 0){
    std::cout << "init output tensor map failed" << std::endl;
    return -5;
  }
  return 0;
}

int32_t SuperPointExcutor::setInputImages(const cv::Mat& input){
  void* host_buffer = this->buffer_manager_ptr_->getHostBuffer(input_tensor_names_[0]);
  if(host_buffer == nullptr){
    std::cout << "[SuperPointExcutor]get host buffer failed" << std::endl;
    return -1;
  }
  memcpy(this->buffer_manager_ptr_->getHostBuffer(input_tensor_names_[0]),
    input.data, this->input_tensor_map_["image"].binding_size);
  buffer_manager_ptr_->copyInputToDeviceAsync(stream_);
  return 0;
}

int32_t SuperPointExcutor::doInference(){
  bool status = this->nv_context_ptr_->enqueueV2(buffer_manager_ptr_->getDeviceBindings().data(), stream_, nullptr);
  if (!status) {
    std::cout << "enqueueV2 failed" << std::endl;
    return -1;
  }
  return 0;
}

int32_t SuperPointExcutor::copyBack(){
  buffer_manager_ptr_->copyOutputToHostAsync(stream_);
  return 0;
}

int32_t SuperPointExcutor::synchronize(){
  cudaStreamSynchronize(stream_);
  return 0;
}

int32_t SuperPointExcutor::getOutput(float* semi_out_put_ptr, float* desc_out_put_ptr){
  auto output_tensor = this->buffer_manager_ptr_->getHostBuffer(output_tensor_names_[0]);
  memcpy(semi_out_put_ptr, output_tensor, this->output_tensor_map_["semi"].binding_size);
  output_tensor = this->buffer_manager_ptr_->getHostBuffer(output_tensor_names_[1]);
  memcpy(desc_out_put_ptr, output_tensor, this->output_tensor_map_["desc"].binding_size);
  return 0;
}

int32_t SuperPointExcutor::initTensorMap(std::vector<std::string> tensor_names,
    TensorMap & tensor_map){
  for(int32_t i = 0; i < tensor_names.size(); i++){
    auto index = this->engine_ptr_->getBindingIndex(tensor_names[i].c_str());
    if(index < 0){
      std::cout << "get input binding index failed" << std::endl;
      return -4;
    }
    auto input_dim = this->engine_ptr_->getBindingDimensions(index);
    int32_t size = input_dim.d[0] * input_dim.d[1] * input_dim.d[2] * input_dim.d[3] *sizeof(float);
    if(size < 0){
      std::cout << "get input binding size failed" << std::endl;
      return -5;
    }
    NodeInfo node_info;
    node_info.tensor_name = tensor_names[i];
    node_info.binding_index = index;
    node_info.binding_size = size;
    tensor_map[tensor_names[i]] = node_info;
  }
  return 0;
}

int32_t SuperPointExcutor::getTensorSize(std::string & tensor_name){
  auto index = this->engine_ptr_->getBindingIndex(tensor_name.c_str());
  if(index < 0){
    std::cout << "get input binding index failed" << std::endl;
    return -1;
  }
  auto input_dim = this->engine_ptr_->getBindingDimensions(index);
  int32_t size = input_dim.d[0] * input_dim.d[1] * input_dim.d[2] * input_dim.d[3] *sizeof(float);
  if(size < 0){
    std::cout << "get input binding size failed" << std::endl;
    return -2;
  }
  return size;
}



}//namespace TensorRTSupperPoint