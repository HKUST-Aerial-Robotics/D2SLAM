#pragma once
#include "onnx_generic.h"
#include "../utils.h"
#include "d2common/utils.hpp"

#define NETVLAD_DESC_RAW_SIZE 4096
namespace D2FrontEnd {
using D2Common::Utility::TicToc;
class MobileNetVLADONNX: public ONNXInferenceGeneric {
protected:
    std::array<float, NETVLAD_DESC_RAW_SIZE> results_;
    std::array<int64_t, 2> output_shape_;
    std::array<int64_t, 4> input_shape_;
    Eigen::MatrixXf pca_comp_T;
    Eigen::VectorXf pca_mean;
public:
    const int descriptor_size = 4096;
    MobileNetVLADONNX(std::string engine_path, int _width, int _height, bool use_tensorrt = true, 
                bool use_fp16 = true, bool use_int8 = false, std::string int8_calib_table_name = ""): 
            ONNXInferenceGeneric(engine_path, "image:0", "descriptor:0", _width, _height, 
                    use_tensorrt, use_fp16, use_int8, int8_calib_table_name),
            output_shape_{1, NETVLAD_DESC_RAW_SIZE},
            input_shape_{1, _height, _width, 1},
            results_{0}
    {
        std::cout << "Trying to init MobileNetVLADONNX@" << engine_path << 
            " tensorrt " << use_tensorrt << " fp16 " << use_fp16 << " int8 " << use_int8 << 
            " pca " << params->enable_pca_netvlad << std::endl;
        input_image = new float[width*height];
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
               input_image, width*height, input_shape_.data(), 4);
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
            results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
        if (params->enable_pca_netvlad) {
            printf("[D2FrontEnd] Loading PCA for MobileNetVLADONNX: %s\n", params->pca_netvlad.c_str());
            auto pca = load_csv_mat_eigen(params->pca_netvlad);
            // This first row is the mean
            pca_mean = pca.row(0).transpose();
            // // The rest is the components
            pca_comp_T = pca.block(1, 0, pca.rows() - 1, pca.cols());
            printf("PCA Comp shape %d x %d mean %d\n", pca_comp_T.rows(), pca_comp_T.cols(), pca_mean.size());
        } else {
            pca_comp_T.resize(0, 0);
            pca_mean.resize(0);
        }
    }

    std::vector<float> inference(const cv::Mat & input) {
        TicToc tic;
        cv::Mat _input;
        if (input.channels() == 3) {
            cv::cvtColor(input, _input, cv::COLOR_BGR2GRAY);
        } else {
            _input = input;
        }
        if (_input.rows != height || _input.cols != width) {
            cv::resize(_input, _input, cv::Size(width, height));
        } 
        _input.convertTo(_input, CV_32F); // DO NOT SCALING HERE
        doInference(_input.data, 1);
        if (params->enable_perf_output) {
            printf("MobileNetVLADONNX::inference() took %f ms\n", tic.toc());
        }
        // Perform PCA if neccasary
        if (pca_comp_T.rows() > 0) {
            Eigen::Map<Eigen::VectorXf> desc(results_.data(), results_.size());
            Eigen::VectorXf desc_pca = pca_comp_T * (desc - pca_mean);
            // Normalize and return
            desc_pca /= desc_pca.norm();
            return std::vector<float>(desc_pca.data(), desc_pca.data() + desc_pca.size());
        }
        return std::vector<float>(results_.begin(), results_.end());
    }
};
}
