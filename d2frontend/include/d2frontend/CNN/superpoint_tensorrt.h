#pragma once

#include <vector>
#include <memory>

#ifdef USE_CUDA
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "tensorrt_utils/buffers.h"
#endif

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "d2frontend/utils.h"

namespace D2FrontEnd {
  struct SuperPointConfig {
  int32_t max_keypoints = 100;
  int32_t remove_borders = 1;
  int32_t dla_core;
  int32_t fp_16;
  int32_t input_width;
  int32_t input_height;
  int32_t superpoint_pca_dims = -1;
  float keypoint_threshold = 0.015;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_path;
  std::string engine_path;
  std::string pca_mean_path;
  std::string pca_comp_path;
  bool enable_pca;
};

#ifdef USE_CUDA
class  SuperPoint {
public:

  explicit SuperPoint(const SuperPointConfig &super_point_config);
  bool build();
  // bool infer(const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &features);
  bool infer(const cv::Mat &image, 
    std::vector<Eigen::Vector2f> &keypoints, 
    std::vector<Eigen::VectorXf> &descriptors,
    std::vector<float> & scores);
  
  bool infer(const cv::Mat & input, std::vector<cv::Point2f> & keypoints,
    std::vector<float> & local_descriptors, std::vector<float> & scores); //a middle level for D2SLAM
  
  void saveEngine();
  bool deserializeEngine();
  void visualization(const std::string &image_name, const cv::Mat &image,std::vector<Eigen::Vector2f> &keypoints);
private:
  SuperPointConfig super_point_config_;
  nvinfer1::Dims input_dims_{};
  nvinfer1::Dims semi_dims_{};
  nvinfer1::Dims desc_dims_{};
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::vector<int>> keypoints_;
  std::vector<std::vector<float>> descriptors_;
  Eigen::MatrixXf pca_comp_T_;
  Eigen::RowVectorXf pca_mean_;

  bool constructNetwork(tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                          tensorrt_common::TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                          tensorrt_common::TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                          tensorrt_common::TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

  bool processInput(const tensorrt_buffer::BufferManager &buffers, const cv::Mat &image);

  // bool processOutput(const tensorrt_buffer::BufferManager &buffers, Eigen::Matrix<float, kSuperPointDescDim, Eigen::Dynamic> &features);
  
  bool processOutput(const tensorrt_buffer::BufferManager &buffers, std::vector<Eigen::Vector2f> &keypoints,
                        std::vector<Eigen::VectorXf>  &descriptors, std::vector<float> &scores);

  void removeBorders( std::vector<Eigen::Vector2f> &keypoints, std::vector<float> &scores, int border, int height,
                      int width);

  std::vector<size_t> sortIndexes(std::vector<float> &data);

  void topKeypoints( std::vector<Eigen::Vector2f> &keypoints, std::vector<float> &scores, int k);

  void findHighScoreIndex(std::vector<float> &scores,  std::vector<Eigen::Vector2f> &keypoints, int h, int w,
                              float threshold);

  void sampleDescriptors( std::vector<Eigen::Vector2f> &keypoints, float *des_map,
                          std::vector<Eigen::VectorXf> &descriptors, int dim, int h, int w, int s = 8);
};

#endif

} // namespace D2FrontEnd
