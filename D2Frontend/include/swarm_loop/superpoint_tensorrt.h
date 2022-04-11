#pragma once

#ifdef USE_TENSORRT
#include "swarm_loop/tensorrt_generic.h"
#include <torch/csrc/autograd/variable.h>
#include <ATen/ATen.h>
#include <torch/csrc/api/include/torch/types.h>
#include <Eigen/Dense>

#define SP_DESC_RAW_LEN 256
namespace Swarm {
class SuperPointTensorRT: public TensorRTInferenceGeneric {
    Eigen::MatrixXf pca_comp_T;
    Eigen::RowVectorXf pca_mean;

public:
    double thres = 0.015;
    bool enable_perf;
    int max_num = 200;
    SuperPointTensorRT(std::string engine_path, 
        std::string _pca_comp,
        std::string _pca_mean,
        int _width, int _height, float _thres = 0.015, int _max_num = 200, bool _enable_perf = false);

    void getKeyPoints(const cv::Mat & prob, float threshold, std::vector<cv::Point2f> &keypoints);
    void computeDescriptors(const torch::Tensor & mProb, const torch::Tensor & desc, const std::vector<cv::Point2f> &keypoints, std::vector<float> & local_descriptors);

    void inference(const cv::Mat & input, std::vector<cv::Point2f> & keypoints, std::vector<float> & local_descriptors);
};
}
#endif