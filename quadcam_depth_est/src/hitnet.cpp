#include "../include/hitnet.hpp"
#include <iostream>
#include <unistd.h>

namespace TensorRTHitnet{
//TODO: refine how we create excutor
int32_t HitnetTrt::HitnetTrt::init(const std::string& engine_filepath, int32_t stream_number){
  if(stream_number <= 0){
    stream_number = 1;
  }
  stream_number_ = stream_number;
  //Create runtime
  nv_runtime_ptr_ = nvinfer1::createInferRuntime(this->logger_);
  if (nv_runtime_ptr_ == nullptr){
    std::cout << "createInferRuntime failed" << std::endl;
    return -2;
  }

  //Create Engine
  if(engine_filepath.empty()){
    std::cout << "engine_filepath is empty" << std::endl;
    return -1;
  }
  std::ifstream engine_file(engine_filepath, std::ios::binary);
  if(!engine_file){
    std::cout << "open engine file failed" << std::endl;
    return -2;
  }
  std::stringstream engine_buffer;
  engine_buffer << engine_file.rdbuf();
  std::string plan = engine_buffer.str();
  printf("engine plan size: %d\n", plan.size());
  auto engine = this->nv_runtime_ptr_->deserializeCudaEngine(plan.data(), plan.size(), nullptr);
  if(engine == nullptr){
    std::cout << "deserializeCudaEngine failed" << std::endl;
    return -3;
  }
  std::shared_ptr<nvinfer1::ICudaEngine> nv_engine(engine, InferDeleter());
  this->nv_engine_ptr_ = nv_engine;
  if(this->nv_engine_ptr_ == nullptr){
    std::cout << "make_shared deserialized Engine failed" << std::endl;
    return -4;
  }

  //Create executors
  for(int32_t i = 0; i < stream_number_; i++){
    HitnetExcutor excutor;
    //TODO: compile bug here
    int32_t ret = excutor.init(this->nv_engine_ptr_,std::string(kInputTensorName),std::string(kOutputTensorName));
    if(ret != 0){
      std::cout << "excutor init failed" << std::endl;
      return -5;
    }
    this->executors_.push_back(std::move(excutor));
  }
  printf("Success to init HitnetTrt\n");
  return 0;
}

//TODO: inference stucked 
int32_t HitnetTrt::doInference(const cv::Mat input[4]){  
  for(int32_t i = 0; i < this->stream_number_; i++){
    if(input[i].empty()){
      return 0;
    }
    int32_t ret = this->executors_[i].setInputImages(input[i]);
    if(ret != 0){
      std::cout << "setInputImages failed" << std::endl;
      return -2;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].doInference();
    if(ret != 0){
      std::cout << "doInference failed" << std::endl;
      return -3;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].copyBack();
    if(ret != 0){
      std::cout << "doInference failed" << std::endl;
      return -3;
    }
  }

  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].synchronize();
    if(ret != 0){
      std::cout << "synchronize failed" << std::endl;
      return -4;
    }
  }

  #ifdef DEBUG
  printf ("inferenced\n");
  #endif

  return 0;
}

int32_t HitnetTrt::getOutput(cv::Mat output[4]){
  for(int32_t i = 0; i < this->stream_number_; i++){
    int32_t ret = this->executors_[i].getOutput(output[i]);
    if(ret != 0){
      std::cout << "getOutput failed" << std::endl;
      return -5;
    }
  }
  return 0;
}

HitnetTrt::~HitnetTrt(){
  for(int32_t i = 0; i < this->stream_number_; i++){
    this->executors_[i].~HitnetExcutor();
  }
  usleep(100);
  this->nv_engine_ptr_->destroy();  //TODO: deconstruct bug here
  usleep(100);
  this->nv_runtime_ptr_->destroy();
  usleep(100);
}

int32_t HitnetExcutor::init(std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr, 
  std::string input_tensor_name,
  std::string output_tensor_name){
  this->engine_ptr_ = engine_ptr;
  this->nv_context_ptr_ = this->engine_ptr_->createExecutionContext();
  if(this->nv_context_ptr_ == nullptr){
    std::cout << "createExecutionContext failed" << std::endl;
    return -1;
  }
  cudaStreamCreate(&this->stream_);
  if(this->stream_ == nullptr){
    std::cout << "cudaStreamCreate failed" << std::endl;
    return -2;
  }
  this->buffer_manager_ptr_ = std::make_unique<CudaMemoryManager::BufferManager>(this->engine_ptr_, 
    1,this->nv_context_ptr_);
  if(this->buffer_manager_ptr_ == nullptr){
    std::cout << "make_unique BufferManager failed" << std::endl;
    return -3;
  }

  this->input_tensor_name_ = input_tensor_name;
  this->output_tensor_name_ = output_tensor_name;
  this->input_index_ = this->engine_ptr_->getBindingIndex(input_tensor_name.c_str());
  if(this->input_index_ < 0){
    std::cout << "getBindingIndex failed" << std::endl;
    return -4;
  }
  auto input_dim = this->engine_ptr_->getBindingDimensions(this->input_index_);
  //Easy Coredump is donnot know your data type
  this->input_size_ = input_dim.d[0] * input_dim.d[1] * input_dim.d[2] * input_dim.d[3] * sizeof(float); 
  this->output_index_ = this->engine_ptr_->getBindingIndex(output_tensor_name.c_str());
  if(this->output_index_  < 0){
    std::cout << "getBindingIndex failed" << std::endl;
    return -5;
  }
  auto output_dim = this->engine_ptr_->getBindingDimensions(this->output_index_ );
  this->output_size_ = output_dim.d[0] * output_dim.d[1] * output_dim.d[2] * output_dim.d[3] * sizeof(float);
  return 0;
}

int32_t HitnetExcutor::setInputImages(const cv::Mat& input){
  memcpy(this->buffer_manager_ptr_->getHostBuffer(this->input_tensor_name_), input.data, this->input_size_);
  buffer_manager_ptr_->copyInputToDeviceAsync(this->stream_);
  return 0;
}

int32_t HitnetExcutor::doInference(){
  bool status = this->nv_context_ptr_->enqueueV2(this->buffer_manager_ptr_->getDeviceBindings().data(), this->stream_, nullptr);
  if (!status){
    std::cout << "enqueueV2 failed" << std::endl;
    return -1;
  }
  return 0;
}

int32_t HitnetExcutor::copyBack(){
  buffer_manager_ptr_->copyOutputToHostAsync(this->stream_);
  return 0;
}

int32_t HitnetExcutor::synchronize(){
  return cudaStreamSynchronize(this->stream_);
}

int32_t HitnetExcutor::getOutput(cv::Mat& output){
  memcpy(output.data, this->buffer_manager_ptr_->getHostBuffer(this->output_tensor_name_), this->output_size_);
  return 0;
}
}