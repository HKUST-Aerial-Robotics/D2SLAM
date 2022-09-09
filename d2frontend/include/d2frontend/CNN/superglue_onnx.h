#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace D2FrontEnd {
class SuperGlueOnnx {
    const int64_t dim_desc = 256;
    OrtMemoryInfo* memory_info=nullptr;
    Ort::Env env;
    Ort::Session * session_ = nullptr;
    const char* input_names[6] {"descriptors0", "keypoints0", "scores0", "descriptors1", "keypoints1", "scores1"};
    const char* output_names[4] {"matches0", "matches1", "matches_scores0", "matches_scores1"};

    void init(const std::string & engine_path);
    void resetTensors();
public:
    SuperGlueOnnx(const std::string & engine_path);
    virtual std::vector<cv::DMatch> inference(const std::vector<cv::Point2f> kpts0, const std::vector<cv::Point2f> kpts1, 
        const std::vector<float> & desc0, const std::vector<float> & desc1, const std::vector<float> & scores0, const std::vector<float> & scores1);
};
}
// #endif