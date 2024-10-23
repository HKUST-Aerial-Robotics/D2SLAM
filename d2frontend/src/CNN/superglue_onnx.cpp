#include <d2frontend/CNN/superglue_onnx.h>
#include <d2frontend/CNN/superpoint_common.h>

namespace D2FrontEnd {
std::vector<float> flatten(const std::vector<cv::Point2f>& vec) {
  std::vector<float> res;
  for (const auto& p : vec) {
    res.push_back(p.x);
    res.push_back(p.y);
  }
  return res;
}
SuperGlueOnnx::SuperGlueOnnx(const std::string& engine_path)
    : env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SuperGlueOnnx")) {
  init(engine_path);
}

void SuperGlueOnnx::init(const std::string& engine_path) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  OrtCUDAProviderOptions options;
  options.device_id = 0;
  options.arena_extend_strategy = 0;
  options.gpu_mem_limit = 1 * 1024 * 1024 * 1024;
  options.cudnn_conv_algo_search =
      OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
  options.do_copy_in_default_stream = 1;
  session_options.AppendExecutionProvider_CUDA(options);
  printf("[SuperGlueOnnx] Loading superglue from %s...\n", engine_path.c_str());
  session_ = new Ort::Session(env, engine_path.c_str(), session_options);
  memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
}

void SuperGlueOnnx::resetTensors() {}

std::vector<cv::DMatch> SuperGlueOnnx::inference(
    const std::vector<cv::Point2f> kpts0, const std::vector<cv::Point2f> kpts1,
    const std::vector<float>& desc0, const std::vector<float>& desc1,
    const std::vector<float>& scores0, const std::vector<float>& scores1) {
  std::vector<cv::DMatch> matches;
  if (kpts0.size() == 0 || kpts1.size() == 0) {
    return matches;
  }
  int num_kpts0 = kpts0.size();
  int num_kpts1 = kpts1.size();
  if (memory_info != nullptr) {
    resetTensors();
  }
  std::array<int64_t, 3> input_desc_dim0{1, num_kpts0, dim_desc};
  std::array<int64_t, 3> input_kps_dim0{1, num_kpts0, 2};
  std::array<int64_t, 2> input_scores_dim0{1, num_kpts0};
  std::array<int64_t, 3> input_desc_dim1{1, num_kpts1, dim_desc};
  std::array<int64_t, 3> input_kps_dim1{1, num_kpts1, 2};
  std::array<int64_t, 2> input_scores_dim1{1, num_kpts1};
  std::vector<Ort::Value> inputs;
  std::vector<float> _kpts0 = flatten(kpts0);
  std::vector<float> _kpts1 = flatten(kpts1);
  // printf("num_kpts0 %ld num_kpts1 %ld\n", num_kpts0, num_kpts1);
  // printf("desc0 data size %ld\n", desc0.size());
  // printf("desc1 data size %ld\n", desc1.size());
  // printf("scores0 data size %ld\n", scores0.size());
  // printf("scores1 data size %ld\n", scores1.size());
  // printf("kpts0 data size %ld\n", _kpts0.size());
  // printf("kpts1 data size %ld\n", _kpts1.size());
  // std::cout << std::endl;
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(desc0.data()), dim_desc * num_kpts0,
      input_desc_dim0.data(), input_desc_dim0.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, _kpts0.data(), 2 * num_kpts0, input_kps_dim0.data(),
      input_kps_dim0.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(scores0.data()), num_kpts0,
      input_scores_dim0.data(), input_scores_dim0.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(desc1.data()), dim_desc * num_kpts1,
      input_desc_dim1.data(), input_desc_dim1.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, _kpts1.data(), 2 * num_kpts1, input_kps_dim1.data(),
      input_kps_dim1.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(scores1.data()), num_kpts1,
      input_scores_dim1.data(), input_scores_dim1.size()));
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names,
                               inputs.data(), 6, output_names, 4);
  for (int i = 0; i < num_kpts0; i++) {
    auto id_j = outputs[0].GetTensorData<int64_t>()[i];
    if (id_j < 0) {
      continue;
    }
    auto score = outputs[2].GetTensorData<float>()[i];
    cv::DMatch match;
    match.queryIdx = i;
    match.trainIdx = id_j;
    match.distance = 1.0 - score;
    matches.emplace_back(match);
    // printf("i->id_j %d->%d\n", i, id_j);
    // printf("[SuperGlueOnnx] Match %d -> %d, score %f\n", i, id_j, score);
  }
  return matches;
}
}  // namespace D2FrontEnd