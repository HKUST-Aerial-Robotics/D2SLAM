#include "tensorrt_basic.hpp"
#include <fstream>
#include <vector>
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace TensorRTBasic{

int32_t TensorRTEngine::initEngine(const std::string& engine_path, 
  const std::string& quat_calib_path, bool enable_fp16, bool enable_int8){
    this->engine_path_ = engine_path;
    this->quat_calib_path_ = quat_calib_path;
  this->nv_runtime_  = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(this->logger_));
  if (this->nv_runtime_ == nullptr){
    printf("[TensorRTBasic][ERROR] nv runtime init failed\n");
    return -1;
  }
  std::ifstream engineFile(engine_path, std::ios::binary);
  if (!engineFile){
    printf("[TensorRTBasic][ERROR] engine file:%s  not found\n", engine_path.c_str());
    return -2;
  }
  engineFile.seekg(0, std::ios::end);
  size_t size = engineFile.tellg();
  engineFile.seekg(0, std::ios::beg);
  std::vector<char> engineData(size);
  engineFile.read(engineData.data(), size);
  this->nv_engine_ = std::make_shared<nvinfer1::ICudaEngine>(this->nv_runtime_->deserializeCudaEngine(engineData.data(), size, nullptr));
  if (this->nv_engine_ == nullptr){
    printf("[TensorRTBasic][ERROR] engine deserialization failed\n");
    return -3;
  }
  printf("[TensorRTBasic][INFO] engine:%s deserialization success\n", engine_path.c_str());
  return 0;
}

std::shared_ptr<nvinfer1::ICudaEngine> TensorRTEngine::getEngine(){
  return this->nv_engine_;
}

TensorRTExcutor::TensorRTExcutor(){
  this->context_ = nullptr;
  this->stream_ = nullptr;
}

TensorRTExcutor::~TensorRTExcutor(){
  if(this->context_ != nullptr){
    this->context_->destroy();
  }
  if(this->stream_ != nullptr){
    cudaStreamDestroy(this->stream_);
  }
}

int32_t TensorRTExcutor::initContexAndStream(std::shared_ptr<nvinfer1::ICudaEngine>& engine){
  cudaError_t cuda_err = cudaSuccess;
  this->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
  if(this->context_ == nullptr){
    printf("[TensorRTExcutor][ERROR] context is nullptr\n");
    return -1;
  }
  cuda_err = cudaStreamCreate(&this->stream_);
  if(cuda_err != cudaSuccess){
    printf("[TensorRTExcutor][ERROR] cuda stream create failed\n");
    return -2;
  }
  return 0;
}


}
