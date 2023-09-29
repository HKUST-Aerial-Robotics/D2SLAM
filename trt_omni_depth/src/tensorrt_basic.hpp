#include <iostream>
#include <stdio.h>
#include <string>
#include <list>
#include <NvInfer.h>
#include <memory>

namespace TensorRTBasic{
class TensorRTEngine{
 public:
  TensorRTEngine(){};
  ~TensorRTEngine(){};
  int32_t initEngine(const std::string& engine_path, const std::string& quat_calib_path, 
    bool enable_fp16, bool enable_int8); //TODO: quat calibration not enabled
  std::shared_ptr<nvinfer1::ICudaEngine> getEngine();
 protected:
  std::string engine_path_;
  std::string quat_calib_path_;
  std::unique_ptr<nvinfer1::IRuntime>nv_runtime_= nullptr;
  std::shared_ptr<nvinfer1::ICudaEngine> nv_engine_ = nullptr;
  std::vector<TensorRTExcutor> excutor_list_;
  Logger logger_;
};


class TensorRTExcutor{
 public:
  TensorRTExcutor();
  ~TensorRTExcutor();
  int32_t initContexAndStream(std::shared_ptr<nvinfer1::ICudaEngine>& engine);
  virtual int32_t setInputData()=0;
  virtual int32_t doInferrence()=0;
  virtual int32_t getOutputData()=0;
 protected:
  std::unique_ptr<nvinfer1::IExecutionContext> context_; //excution context
  cudaStream_t stream_;//cuda stream
};

class Logger: public nvinfer1::ILogger{
  void log(Severity severity, const char* msg) noexcept override{
    if(severity <= Severity::kINFO){
      std::cout << msg <<std::endl;
    }
  }
};

}


