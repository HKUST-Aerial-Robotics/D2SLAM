#include "trt_buffers.h"
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "CNN_generic.h"
#include "d2common/d2frontend_types.h"

namespace TensorRTSupperPoint{

typedef std::map<std::string, NodeInfo> TensorMap;
const int32_t kSpuerpointDesRawLen = 256;

class SuperPointLogger: public nvinfer1::ILogger{
  void log(Severity severity, const char* msg) noexcept override{
    if(severity <= Severity::kWARNING){
      std::cout << msg <<std::endl;
    }
  }
};

class SuperPointExcutor{
public:
  SuperPointExcutor(){};
  ~SuperPointExcutor(){
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
  int32_t getOutput(float* semi_out_put_ptr, float* desc_out_put_ptr);
  int32_t getTensorSize(std::string & tensor_name);

private:
  int32_t initTensorMap(std::vector<std::string> input_tensor_names,
    TensorMap & tensor_map);
  std::shared_ptr<nvinfer1::ICudaEngine> engine_ptr_ = nullptr;
  nvinfer1::IExecutionContext* nv_context_ptr_ = nullptr;
  std::shared_ptr <CudaMemoryManager::BufferManager> buffer_manager_ptr_; //TODO: risky
  cudaStream_t stream_;
  std::vector<std::string> input_tensor_names_ = {"image"};
  std::vector<std::string> output_tensor_names_ = {"semi", "desc"};
  TensorMap input_tensor_map_;
  TensorMap output_tensor_map_;
};

class SuperPointTrt{
public:
  SuperPointTrt(int32_t stream_number,int32_t width, 
    int32_t height, int32_t min_dist, double thres, int32_t max_num, std::string & engine_path,
    std::string& pca_comp_path, std::string& pca_mean_path, bool show_info);
  ~SuperPointTrt(){};
  int32_t init();
  int32_t doInference(std::vector<cv::Mat>& inputs);
  int32_t getOuput(const int drone_id, const D2Common::StereoFrame & msg, D2Common::VisualImageDescArray& desc_array);
private:
  nvinfer1::IRuntime* nv_runtime_ptr_ = nullptr;
  std::shared_ptr<nvinfer1::ICudaEngine> nv_engine_ptr_ = nullptr;
  SuperPointLogger logger_;
  std::vector<SuperPointExcutor> executors_;
  std::vector<float*> semi_out_put_ptr_;
  std::vector<float*> desc_out_put_ptr_; 
  std::vector<std::string> input_tensor_names_ = {"image"};
  std::vector<std::string> output_tensor_names_ = {"semi", "desc"};
  
  Eigen::MatrixXf pca_comp_T;
  Eigen::RowVectorXf pca_mean;
  std::string pca_comp_path_;
  std::string pca_mean_path_;
  std::string engine_path_;

  int32_t stream_number_ = 0;
  int32_t height_ = 300;
  int32_t width_ = 150;
  int max_num_ = 200;
  int nms_dist_ = 5;
  double thres_ = 0.015;
  bool show_info_;

};
}