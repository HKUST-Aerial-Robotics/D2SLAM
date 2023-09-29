#include "tensorrt_basic.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

namespace HitNetTrt{
class HitNet{
 public:
  HitNet(){};
  ~HitNet(){};
  int32_t initHitNet(const std::string& engine_path, const std::string& quat_calib_path = "",
    int32_t infer_streams =1,bool enable_fp16 = true, bool enable_int8 = false);
  int32_t setInputData(std::vector<cv::Mat>& stereo_pair_vec);
  int32_t doInferrence();
  int32_t getOutputs(std::vector<cv::Mat>& depth_estimation_vec);
 private:
  TensorRTBasic::TensorRTEngine hitnet_engine_;
  std::list<std::unique_ptr<HitNetExcutor>> excutor_ptr_list_;
};


class HitNetExcutor:public TensorRTBasic::TensorRTExcutor{
 public:
  HitNetExcutor();
  ~HitNetExcutor();
  int32_t setInputData(cv::Mat& stereo_pair);
  int32_t doInferrence();
  int32_t getOutputData(cv::Mat& depth);
 private:
#if JETSON
 //TODO: unified memory
#else
  cv::Mat input_mat_;
  cv::Mat output_mat_;
#endif
};

}