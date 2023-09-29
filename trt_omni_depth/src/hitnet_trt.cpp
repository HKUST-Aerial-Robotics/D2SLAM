#include "hitnet_trt.hpp"

namespace HitNetTrt{

int32_t HitNet::initHitNet(const std::string& engine_path, const std::string& quat_calib_path, 
  int32_t infer_streams,bool enable_fp16, bool enable_int8){
  printf("[Init TensorRT Engine from file]: %s\n", engine_path.c_str());
  int32_t ret = -1;
  ret = hitnet_engine_.initEngine(engine_path, quat_calib_path, enable_fp16, enable_int8);
  if (ret != 0){
    printf("[ERROR] Init TensorRT Engine from file failed!\n");
    return -1;
  }
  for (int i = 0; i < infer_streams ; i++){
    auto new_excutor_ptr = std::make_unique<HitNetExcutor>();
    auto engine_ptr = hitnet_engine_.getEngine();
    ret = new_excutor_ptr->initContexAndStream(engine_ptr);
    if (ret != 0){
      printf("[ERROR] Init TensorRT Excutor failed!\n");
      return -1;
    }
    excutor_ptr_list_.push_back(std::move(new_excutor_ptr));
  }
  return 0;
}

int32_t HitNet::setInputData(std::vector<cv::Mat>& stereo_pair_vec){
  int i = 0 ;
  for (auto && iter : this->excutor_ptr_list_){
    iter->setInputData(stereo_pair_vec[i++]);
  }
  return 0;
}

int32_t HitNet::doInferrence(){
  for (auto && iter : this->excutor_ptr_list_){
    iter->doInferrence();
  }
  return 0;
}

int32_t HitNet::getOutputs(std::vector<cv::Mat>& depth_estimation_vec){
  int i = 0 ;
  for (auto && iter : this->excutor_ptr_list_){
    iter->getOutputData(depth_estimation_vec[i++]);
  }
  return 0;
}

HitNetExcutor::HitNetExcutor(){
  contex_ = nullptr;
  stream_ = nullptr;
}
HitNetExcutor::~HitNetExcutor(){
  if (contex_ != nullptr){
    contex_->destroy();
  }
  if (stream_ != nullptr){
    cudaStreamDestroy(stream_);
  }
}

int32_t HitNetExcutor::setInputData(cv::Mat& stereo_pair){
  stereo_pair.copyTo(input_mat_);
  return 0;
}
int32_t HitNetExcutor::doInferrence(){
  return 0;
}
int32_t HitNetExcutor::getOutputData(cv::Mat& depth){
  return 0;
}
}
