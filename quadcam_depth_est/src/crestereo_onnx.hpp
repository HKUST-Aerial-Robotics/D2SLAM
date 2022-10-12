#pragma once
#include <d2frontend/CNN/onnx_generic.h>
#include <d2common/utils.hpp>

using D2Common::Utility::TicToc;

namespace D2QuadCamDepthEst {
// inline void hwc_to_chw(cv::Mat src, cv::Mat & dst) {
//     const int src_h = src.rows;
//     const int src_w = src.cols;
//     const int src_c = src.channels();
//     cv::Mat hw_c = src.reshape(1, src_h * src_w);
//     // const std::array<int,3> dims = {src_c, src_h, src_w};                         
//     // dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));                         
//     cv::Mat dst_1d = dst.reshape(1, {src_c, src_h, src_w});              
//     cv::transpose(hw_c, dst_1d);                                                  
// }  

inline void hwc_to_chw(const cv::Mat & src, float * buf) {
    const int src_h = src.rows;
    const int src_w = src.cols;
    const int src_c = src.channels();
    const int aera = src_h * src_w;
    if (src.type() == CV_8UC3) {
        for (int i = 0; i < src_h; i++) {
            for (int j = 0; j < src_w; j++) {
                const cv::Vec3b bgr = src.at<cv::Vec3b>(i, j);
                buf[i * src_w + j] = bgr[0]; //Red
                buf[aera + i * src_w + j] = bgr[1]; //Green
                buf[aera*2 + i * src_w + j] = bgr[2]; //Blue
            }
        }
    }
    if (src.type() == CV_32FC3) {
        for (int i = 0; i < src_h; i++) {
            for (int j = 0; j < src_w; j++) {
                const cv::Vec3f bgr = src.at<cv::Vec3f>(i, j);
                buf[i * src_w + j] = bgr[0]; //Red
                buf[aera + i * src_w + j] = bgr[1]; //Green
                buf[aera*2 + i * src_w + j] = bgr[2]; //Blue
            }
        }
    }

}  

class CREStereoONNX: public D2FrontEnd::ONNXInferenceGeneric {
protected:
    float * results_;
    std::array<int64_t, 4> output_shape_;
    std::array<int64_t, 4> input_shape;
    std::array<int64_t, 4> input_shape_half;
    float* input_l, *input_r, * input_l_half, *input_r_half;
    std::vector<Ort::Value> inputs;
    bool combined = false;
    Ort::MemoryInfo memory_info;
    void setInputs(Ort::MemoryInfo & memory_info, int width, int height) {
        inputs.clear();
        const std::array<int,3> dims = {3, height, width};                         
        input_l = new float[width*height*3];
        input_r = new float[width*height*3];
        if (combined) {
            const std::array<int,3> dims = {3, height/2, width/2};
            input_l_half = new float[width/2*height/2*3];
            input_r_half = new float[width/2*height/2*3];
            inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
                    input_l_half, 3*width*height/4, input_shape_half.data(), 4));
            inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
                   input_r_half, 3*width*height/4, input_shape_half.data(), 4));
        }
        inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
               input_l, 3*width*height, input_shape.data(), 4));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
               input_r, 3*width*height, input_shape.data(), 4));
    }
public:
    CREStereoONNX(std::string engine_path, int _width, int _height, bool use_tensorrt = true, bool use_fp16 = true, bool use_int8 = false): 
            ONNXInferenceGeneric(engine_path, "left", "output", _width, _height, use_tensorrt, use_fp16, use_int8),
            output_shape_{1, 2, _height, _width}, //Output shape is 2 channels, height, width
            input_shape{1, 3, _height, _width}, //CREStereo is batch channel height width
            input_shape_half{1, 3, _height/2, _width/2}, //CREStereo is batch channel height width
            memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        std::cout << "Trying to init CREStereoONNX@" << engine_path << std::endl;
        if (session_->GetInputCount() == 2) {
            combined = false;
        } else {
            combined = true;
        }
        setInputs(memory_info, _width, _height);
        results_ = new float[2*width*height];
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
            results_, 2*width*height, output_shape_.data(), output_shape_.size());

        if (!combined) {
            printf("CREStereoONNX Init initialized\n");
        } else {
            printf("CREStereoONNX Combine initialized\n");
        }
    }

    virtual void doInference() {
        const char* input_names[] = {"init_left", "init_right", "next_left", "next_right"};
        const char* output_names[] = {"next_output"};
        if (combined) {
            session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 4, output_names, &output_tensor_, 1);
        } else {
            input_names[0] = "left";
            input_names[1] = "right";
            output_names[0] = "output";
            session_->Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 2, output_names, &output_tensor_, 1);
        } 
    }

    cv::Mat inference(const cv::Mat & input_left, const cv::Mat & input_right) {
        cv::Mat _input_left, _input_right;
        if (input_left.channels() != 3) {
            cv::cvtColor(input_left, _input_left, cv::COLOR_GRAY2RGB);
            cv::cvtColor(input_right, _input_right, cv::COLOR_GRAY2RGB);
        } else {
            _input_left = input_left;
            _input_right = input_right;
        }
        if (_input_left.rows != height || _input_left.cols != width) {
            cv::resize(_input_left, _input_left, cv::Size(width, height));
            cv::resize(_input_right, _input_right, cv::Size(width, height));
        } 
        cv::Mat input_half_left, input_half_right;
        cv::resize(_input_left, input_half_left, cv::Size(width/2, height/2));
        cv::resize(_input_right, input_half_right, cv::Size(width/2, height/2));
        hwc_to_chw(_input_left, input_l);
        hwc_to_chw(_input_right, input_r);
        hwc_to_chw(input_half_left, input_l_half);
        hwc_to_chw(input_half_right, input_r_half);
        doInference();
        cv::Mat res(height, width, CV_32F, results_);
        return res;
    }
};
}
