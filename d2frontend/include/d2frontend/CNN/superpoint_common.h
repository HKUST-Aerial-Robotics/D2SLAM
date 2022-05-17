#pragma once
#include <torch/csrc/autograd/variable.h>
#include <ATen/ATen.h>
#include <torch/csrc/api/include/torch/types.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#define SP_DESC_RAW_LEN 256
#define USE_PCA

namespace D2FrontEnd {
void getKeyPoints(const cv::Mat & prob, float threshold, std::vector<cv::Point2f> &keypoints, int width, int height, int max_num);
void computeDescriptors(const torch::Tensor & mProb, const torch::Tensor & desc,
        const std::vector<cv::Point2f> &keypoints, std::vector<float> & local_descriptors, int width, int height, 
        const Eigen::MatrixXf & pca_comp_T, const Eigen::RowVectorXf & pca_mean);
}