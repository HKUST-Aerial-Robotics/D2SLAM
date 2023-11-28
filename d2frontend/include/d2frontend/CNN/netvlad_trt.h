#pragma once
#include "trt_buffers.h"
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include "CNN_generic.h"
#include "d2common/d2frontend_types.h"
#include <vector>
#include <array>

namespace TensorRTNetVLAD{
const int32_t kNetVLADDesRawLen = 4096;
typedef std::map<std::string, NodeInfo> TensorMap;

class NetVLADLogger: public nvinfer1::ILogger{
  void log(Severity severity, const char* msg) noexcept override{
    if(severity <= Severity::kWARNING){
      std::cout << msg <<std::endl;
    }
  }
};

class NetVLADExcutor{
public:
  NetVLADExcutor(){};
  ~NetVLADExcutor(){
    if(nv_context_ptr_ != nullptr){
      nv_context_ptr_ = nullptr;
    }
    if(buffer_manager_ptr_ != nullptr){
      buffer_manager_ptr_ = nullptr;
    }
  };

  int32_t init(std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr, 
    std::vector<std::string> input_tensor_names,
    std::vector<std::string> output_tensor_names);
  int32_t setInputImages(const cv::Mat& input);
  int32_t doInference();
  int32_t copyBack();
  int32_t synchronize();
  int32_t getOutput(std::array<float, kNetVLADDesRawLen>& descriptors);
  int32_t getTensorSize(std::string & tensor_name);
private:
  int32_t initTensorMap(std::vector<std::string> input_tensor_names,
    TensorMap & tensor_map);
  std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr_ = nullptr;
  nvinfer1::IExecutionContext* nv_context_ptr_ = nullptr;
  std::shared_ptr <CudaMemoryManager::BufferManager> buffer_manager_ptr_;
  cudaStream_t stream_;
  std::vector<std::string> input_tensor_names_ = {"image:0"};
  std::vector<std::string> output_tensor_names_ = {"descriptor:0"};
  TensorMap input_tensor_map_;
  TensorMap output_tensor_map_;
};

class NetVLADTrt{
public:
  NetVLADTrt(int32_t stream_number, int32_t width, int32_t hight, 
    std::string & engine_path, std::string& pca_table_path, bool show_info);
  ~NetVLADTrt();
  int32_t init();
  int32_t doInference(std::vector<cv::Mat>& inputs);
  int32_t getOutput(std::vector<std::vector<float>> & descriptors);
private:
  nvinfer1::IRuntime* nv_runtime_ptr_ = nullptr;
  std::shared_ptr<nvinfer1::ICudaEngine> nv_engine_ptr_ = nullptr;
  NetVLADLogger logger_;

  std::vector<NetVLADExcutor> excutor_vec_;
  std::vector<std::string> input_tensor_names_ = {"image:0"};
  std::vector<std::string> output_tensor_names_ = {"descriptor:0"};

  std::string engine_path_;
  std::string pca_table_path_;
  std::string pca_comp_path_;
  std::string pca_mean_path_;

  Eigen::MatrixXf pca_comp_T_;
  Eigen::VectorXf pca_mean_;
  
  int32_t stream_number_ = 1;
  int32_t width_ = 640;
  int32_t height_ = 480;
  bool show_info_ = false;

  std::vector<std::array<float, kNetVLADDesRawLen>> descriptors_;
};

}