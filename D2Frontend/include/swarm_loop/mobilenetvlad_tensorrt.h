#pragma once

#ifdef USE_TENSORRT
#include "swarm_loop/tensorrt_generic.h"
namespace Swarm {
class MobileNetVLADTensorRT: public TensorRTInferenceGeneric {
public:
    bool enable_perf;
    const int descriptor_size = 4096;
    MobileNetVLADTensorRT(std::string engine_path, int _width, int _height, bool _enable_perf = false) : TensorRTInferenceGeneric("image:0", _width, _height), enable_perf(_enable_perf) {
        TensorInfo outputTensorDesc;
        outputTensorDesc.blobName = "descriptor:0";
        outputTensorDesc.volume = descriptor_size;
        m_InputSize = height*width;
        m_OutputTensors.push_back(outputTensorDesc);
        std::cout << "Trying to init TRT engine of MobileNetVLADTensorRT" << std::endl;

        init(engine_path);
    }

    std::vector<float> inference(const cv::Mat & input);
};
}
#endif