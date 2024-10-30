#pragma once

#include "onnx_generic.h"
#include <Eigen/Dense>

namespace D2FrontEnd {
class SuperPointONNX: public ONNXInferenceGeneric {
    Eigen::MatrixXf pca_comp_T;
    Eigen::RowVectorXf pca_mean;
    float * results_desc_ = nullptr;
    float * results_semi_ = nullptr;
    std::array<int64_t, 4> output_shape_desc_;
    std::array<int64_t, 3> output_shape_semi_;
    std::array<int64_t, 4> input_shape_;
    std::vector<Ort::Value> output_tensors_;
    int max_num = 200;
    int nms_dist = 10;
public:
    double thres = 0.015;
    SuperPointONNX(std::string engine_path, 
        int _nms_dist,
        std::string _pca_comp,
        std::string _pca_mean,
        int _width, int _height, float _thres = 0.015, int _max_num = 200, bool use_tensorrt = true, 
        bool use_fp16 = true, bool use_int8 = false, std::string int8_calib_table_name = "");

    
    void infer(const cv::Mat & input, std::vector<cv::Point2f> & keypoints, std::vector<float> & local_descriptors, std::vector<float> & scores);
    void doInference(const unsigned char* input, const uint32_t batchSize) override;
};
}