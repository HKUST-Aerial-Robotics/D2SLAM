#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <NvInfer.h>
#include <memory.h>
#include <fstream>
#include <vector>
#include <NvOnnxParser.h>
#include <stdint.h>
#include "tensorrt_utils/buffers.h"

constexpr char kInputTensorName[] = "input";
constexpr char kOutputTensorName[] = "reference_output_disparity";

namespace TensorRTHitnet{

class HitnetLogger: public nvinfer1::ILogger{
  void log(Severity severity, const char* msg) noexcept override{
    if(severity <= Severity::kWARNING){
      std::cout << msg <<std::endl;
    }
  }
};

class HitnetExcutor{
 public:
  HitnetExcutor(){};
  ~HitnetExcutor(){
    if(nv_context_ptr_ != nullptr){
      nv_context_ptr_ = nullptr;
    }
    if(buffer_manager_ptr_ != nullptr){
      buffer_manager_ptr_ = nullptr;
    }
  };

  int32_t init(std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr, 
    std::string input_tensor_name = "input",
    std::string output_tensor_name = "output");
  int32_t setInputImages(const cv::Mat& input);
  int32_t doInference();
  int32_t copyBack();
  int32_t synchronize();
  int32_t getOutput(cv::Mat& output);
  
 private:
  std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr_ = nullptr;
  nvinfer1::IExecutionContext* nv_context_ptr_ = nullptr;
  std::shared_ptr <tensorrt_buffer::BufferManager> buffer_manager_ptr_; //TODO: risky
  cudaStream_t stream_;
  std::string input_tensor_name_ = "input";
  std::string output_tensor_name_ = "output";
  int32_t input_size_ = 0;
  int32_t output_size_ = 0;
  int32_t input_index_ = -1;
  int32_t output_index_ = -1;
};

class HitnetTrt{
 public:
  HitnetTrt(bool show_info):show_info_(show_info){};
  ~HitnetTrt();
  int32_t init(const std::string& onnx_model_path, const std::string& trt_engine_path, int32_t stream_number);
  int32_t doInference(const cv::Mat input[4]);//input  4 1x2x240x320 bcwh output 4 1x1x240x320 disparity
  int32_t getOutput(cv::Mat output[4]);
 private:
  int32_t deserializeEngine(const std::string& trt_engine_path);
  int32_t buildEngine(const std::string& onnx_model_path, const std::string& trt_engine_path);

  int32_t stream_number_ = 4;
  std::vector<HitnetExcutor> executors_;
  std::shared_ptr<nvinfer1::ICudaEngine> nv_engine_ptr_ = nullptr;
  HitnetLogger logger_;
  bool show_info_;
};



}

