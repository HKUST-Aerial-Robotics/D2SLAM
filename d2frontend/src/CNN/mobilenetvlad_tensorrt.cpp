#include "d2frontend/CNN/mobilenetvlad_tensorrt.h"
using namespace D2FrontEnd;

std::vector<float> MobileNetVLADTensorRT::inference(const cv::Mat & input) {
    if (m_Engine == nullptr) {
        // std::cerr  << "MobileNetVLADTensorRT engine failed returing zero vector." << std::endl;
        std::vector<float> ret(descriptor_size, 0.0);
        return ret;
    }
    cv::Mat _input;
    if (input.rows != height || input.cols != width) {
        cv::resize(input, _input, cv::Size(width, height));
        _input.convertTo(_input, CV_32F);
    } else {
        input.convertTo(_input, CV_32F);
    }
    ((CNNInferenceGeneric*) this)->doInference(_input);

    return std::vector<float>(m_OutputTensors[0].hostBuffer, m_OutputTensors[0].hostBuffer+descriptor_size);
}