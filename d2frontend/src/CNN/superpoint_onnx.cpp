#include <d2frontend/CNN/superpoint_common.h>
#include <d2frontend/CNN/superpoint_onnx.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/utils.h>

#include "ATen/Parallel.h"
#include "d2common/utils.hpp"

using D2Common::Utility::TicToc;
namespace D2FrontEnd {
SuperPointONNX::SuperPointONNX(std::string engine_path, int _nms_dist,
                               std::string _pca_comp, std::string _pca_mean,
                               int _width, int _height, float _thres,
                               int _max_num, bool use_tensorrt, bool use_fp16,
                               bool use_int8, std::string int8_calib_table_name)
    : ONNXInferenceGeneric(engine_path, "image", "semi", _width, _height,
                           use_tensorrt, use_fp16, use_int8,
                           int8_calib_table_name),
      output_shape_semi_{1, _height, _width},
      output_shape_desc_{1, SP_DESC_RAW_LEN, _height / 8, _width / 8},
      input_shape_{1, 1, _height, _width},
      thres(_thres),
      max_num(_max_num),
      nms_dist(_nms_dist) {
#ifdef USE_CUDA
  at::set_num_threads(1);
#endif
  std::cout << "Init SuperPointONNX: " << engine_path << " size " << _width
            << " " << _height << std::endl;

  input_image = new float[_width * _height];
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  input_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info, input_image, width * height, input_shape_.data(), 4);

  results_desc_ = new float[1 * SP_DESC_RAW_LEN * height / 8 * width / 8];
  results_semi_ = new float[width * height];
  // semi
  output_tensors_.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, results_semi_, height * width, output_shape_semi_.data(),
      output_shape_semi_.size()));
  // desc
  output_tensors_.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, results_desc_, SP_DESC_RAW_LEN * height / 8 * width / 8,
      output_shape_desc_.data(), output_shape_desc_.size()));
  if (params->enable_pca_superpoint) {
    pca_comp_T = load_csv_mat_eigen(_pca_comp).transpose();
    pca_mean = load_csv_vec_eigen(_pca_mean).transpose();
  } else {
    pca_comp_T.resize(0, 0);
    pca_mean.resize(0);
  }
}

void SuperPointONNX::doInference(const unsigned char* input,
                                 const uint32_t batchSize) {
  const char* input_names[] = {m_InputBlobName.c_str()};
  const char* output_names_[] = {"semi", "desc"};
  memcpy(input_image, input, width * height * sizeof(float));
  session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1,
                output_names_, output_tensors_.data(), 2);
}

void SuperPointONNX::infer(const cv::Mat& input,
                               std::vector<cv::Point2f>& keypoints,
                               std::vector<float>& local_descriptors,
                               std::vector<float>& scores) {
#ifdef USE_CUDA
  TicToc tic;
  cv::Mat _input;
  keypoints.clear();
  local_descriptors.clear();
  assert(input.rows == height && input.cols == width &&
         "Input image must have same size with network");
  if (input.channels() == 3) {
    cv::cvtColor(input, _input, cv::COLOR_BGR2GRAY);
  } else {
    _input = input;
  }
  if (_input.rows != height || _input.cols != width) {
    cv::resize(_input, _input, cv::Size(width, height));
  }
  _input.convertTo(_input, CV_32F, 1 / 255.0);
  ((CNNInferenceGeneric*)this)->doInference(_input);
  double inference_time = tic.toc();

  TicToc tic1;
  auto options = torch::TensorOptions().dtype(torch::kFloat32);

  auto mProb = at::from_blob(results_semi_, {1, 1, height, width}, options);
  auto mDesc = at::from_blob(
      results_desc_, {1, SP_DESC_RAW_LEN, height / 8, width / 8}, options);
  cv::Mat Prob(height, width, CV_32F, results_semi_);
  double copy_time = tic1.toc();

  TicToc tic2;
  getKeyPoints(Prob, thres, nms_dist, keypoints, scores, width, height,
               max_num);
  double nms_time = tic2.toc();
  computeDescriptors(mProb, mDesc, keypoints, local_descriptors, width, height,
                     pca_comp_T, pca_mean);
  double desc_time = tic2.toc();
  if (params->enable_perf_output) {
    printf(
        "[SuperPointONNX] inference time: %f ms, copy time: %f ms, nms time: "
        "%f ms, desc time: %f ms\n",
        inference_time, copy_time, nms_time, desc_time);
  }
#else
  SPDLOG_ERROR("SuperPointONNX not supported without CUDA");
#endif
}
}  // namespace D2FrontEnd
