#pragma once
#include <d2frontend/CNN/onnx_generic.h>
#include <d2common/utils.hpp>

using D2Common::Utility::TicToc;

namespace D2QuadCamDepthEst {
class HitnetONNX: public D2FrontEnd::ONNXInferenceGeneric {
protected:
    float * results_;
    std::array<int64_t, 4> output_shape_;
    std::array<int64_t, 4> input_shape_;
    cv::Mat input_image_mat;
public:
    HitnetONNX(std::string engine_path, int _width, int _height, bool use_tensorrt = true, bool use_fp16 = true, bool use_int8 = false): 
            ONNXInferenceGeneric(engine_path, "input", "reference_output_disparity", _width, _height, use_tensorrt, use_fp16, use_int8),
            output_shape_{1, _height, _width, 1},
            input_shape_{1, 2, _height, _width} {
        std::cout << "Trying to init HitnetONNX@" << engine_path << std::endl;
        input_image_mat = cv::Mat(height*2, width, CV_32F);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
               (float*)input_image_mat.data, 2*width*height, input_shape_.data(), 4);
        results_ = new float[width*height];
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
            results_, width*height, output_shape_.data(), output_shape_.size());
        printf("HitnetONNX initialized\n");
    }

    virtual void doInference() {
        const char* input_names[] = {m_InputBlobName.c_str()};
        const char* output_names[] = {output_name.c_str()};
        session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    }

    cv::Mat inference(const cv::Mat & input_left, const cv::Mat & input_right) {
        TicToc tic;
        cv::Mat _input_left, _input_right;
        if (input_left.channels() == 3) {
            cv::cvtColor(input_left, _input_left, cv::COLOR_BGR2GRAY);
            cv::cvtColor(input_right, _input_right, cv::COLOR_BGR2GRAY);
        } else {
            _input_left = input_left;
            _input_right = input_right;
        }
        if (_input_left.rows != height || _input_left.cols != width) {
            cv::resize(_input_left, _input_left, cv::Size(width, height));
            cv::resize(_input_right, _input_right, cv::Size(width, height));
        } 
        cv::Mat input;
        cv::vconcat(_input_left, _input_right, input);
        input.convertTo(input_image_mat, CV_32F, 1.0/255.0);
        doInference();
        printf("HitnetONNX::inference() took %f ms\n", tic.toc());
        return cv::Mat(height, width, CV_32F, results_);
    }
};
}
