#include "hitnet.hpp"
#include <iostream>
#include <unistd.h>
#include <spdlog/spdlog.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "tensorrt_utils/buffers.h"
#include "tensorrt_utils/logger.h"
#include "tensorrt_utils/common.h"

namespace TensorRTHitnet{
int32_t HitnetTrt::init(const std::string& onnx_model_path, const std::string& trt_engine_path, int32_t stream_number){
  if(stream_number <= 0){
    stream_number = 1;
  }
  stream_number_ = stream_number;

  //Check TRT engine file can be loaded, if not create engine from onnx model and save to trt_engine_path
  if (access(trt_engine_path.c_str(), F_OK) == -1){
    spdlog::info("TRT engine file not found, create engine from onnx model");
    if (access(onnx_model_path.c_str(), F_OK) == -1){
      spdlog::error("onnx model file not found");
      return -2;
    }
    if (buildEngine(onnx_model_path, trt_engine_path) != 0){
      spdlog::error("buildEngine failed");
      return -3;
    }
  } else {
    spdlog::info("TRT engine file found, load engine from file");
    if (deserializeEngine(trt_engine_path) != 0){
      spdlog::error("deserializeEngine failed");
      return -4;
    }
  }

  //Create executors
  for(int32_t i = 0; i < stream_number_; i++){
    HitnetExcutor excutor;
    int32_t ret = excutor.init(this->nv_engine_ptr_,std::string(kInputTensorName),std::string(kOutputTensorName));
    if(ret != 0){
      spdlog::error("Hitnet multistream Excutor init failed");
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
}


int32_t HitnetTrt::deserializeEngine(const std::string& trt_engine_path){
  std::ifstream engine_file(trt_engine_path.c_str(), std::ios::binary);
  if (engine_file.is_open()){
    spdlog::info("load engine from file: {}", trt_engine_path);
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    char * engine_data = new char[size];
    engine_file.read(engine_data, size);
    engine_file.close();
    IRuntime* nv_runtime = createInferRuntime(tensorrt_log::gLogger);
    this->nv_engine_ptr_ = std::shared_ptr<nvinfer1::ICudaEngine>(nv_runtime->deserializeCudaEngine(engine_data, size, nullptr));
    delete[] engine_data;
    if(this->nv_engine_ptr_ == nullptr){
      spdlog::error("deserializeCudaEngine failed");
      return -3;
    }
    spdlog::info("Success to load engine from file");
    return 0;
  } else {
    return -2;
  }
}

int32_t HitnetTrt::buildEngine(const std::string& onnx_model_path, const std::string& trt_engine_path){
  auto builder = tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(tensorrt_log::gLogger.getTRTLogger()));
  if (builder == nullptr){
    spdlog::error("createInferBuilder failed");
    return -1;
  }
  auto network = tensorrt_common::TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
    1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (network == nullptr){
    spdlog::error("createNetworkV2 failed");
    return -2;
  }
  auto config = tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (config == nullptr){
    spdlog::error("createBuilderConfig failed");
    return -3;
  }
  auto parser = tensorrt_common::TensorRTUniquePtr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, tensorrt_log::gLogger.getTRTLogger()));
  if (parser == nullptr){
    spdlog::error("createParser failed");
    return -4;
  }
  auto profile = builder->createOptimizationProfile();
  if (profile == nullptr){
    spdlog::error("createOptimizationProfile failed");
    return -5;
  }
  profile->setDimensions(kInputTensorName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 1, 240, 320});
  config->addOptimizationProfile(profile);

  auto parsed = parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(tensorrt_log::gLogger.getReportableSeverity()));
  if (!parsed){
    spdlog::error("parseFromFile failed");
    return -6;
  }
  config ->setMaxWorkspaceSize(1024_MiB);
  config ->setFlag(nvinfer1::BuilderFlag::kFP16);
  config ->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
  
  auto profile_stream = tensorrt_common::makeCudaStream();
  if (profile_stream == nullptr){
    spdlog::error("makeCudaStream failed");
    return -7;
  }
  config->setProfileStream(*profile_stream);
  tensorrt_common::TensorRTUniquePtr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (plan == nullptr){
    spdlog::error("buildSerializedNetwork failed");
    return -8;
  }
  tensorrt_common::TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(tensorrt_log::gLogger.getTRTLogger())};
  if (runtime == nullptr){
    spdlog::error("createInferRuntime failed");
    return -9;
  }
  this->nv_engine_ptr_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size(), nullptr));
  if (this->nv_engine_ptr_ == nullptr){
    spdlog::error("deserializeCudaEngine failed");
    return -10;
  }

  //Save engine to file
  nvinfer1::IHostMemory* engine_data = this->nv_engine_ptr_->serialize();
  std::ofstream engine_file(trt_engine_path.c_str(), std::ios::binary);
  if (engine_file.is_open()){
    engine_file.write(static_cast<const char*>(engine_data->data()), engine_data->size());
    engine_file.close();
    spdlog::info("Success to save engine to file: {}", trt_engine_path);
  } else {
    spdlog::error("Failed to save engine to file: {}", trt_engine_path);
    return -11;
  }
  return 0;
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
  this->buffer_manager_ptr_ = std::make_unique<tensorrt_buffer::BufferManager>(this->engine_ptr_, 
    0,this->nv_context_ptr_);
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