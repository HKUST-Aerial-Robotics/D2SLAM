#include "swarm_loop/mobilenetvlad_tensorrt.h"
using namespace Swarm;

std::vector<float> MobileNetVLADTensorRT::inference(const cv::Mat & input) {
    cv::Mat _input;
    if (input.rows != height || input.cols != width) {
        cv::resize(input, _input, cv::Size(width, height));
        _input.convertTo(_input, CV_32F);
    } else {
        input.convertTo(_input, CV_32F);
    }
    doInference(_input);

    return std::vector<float>(m_OutputTensors[0].hostBuffer, m_OutputTensors[0].hostBuffer+descriptor_size);
}