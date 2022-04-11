#pragma once

#ifdef USE_TENSORRT

//Original code from https://github.com/enazoe/yolo-tensorrt
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <trt_utils.h>

namespace Swarm {
struct TensorInfo
{
    std::string blobName;
    float* hostBuffer{nullptr};
    uint64_t volume{0};
    int bindingIndex{-1};
};


class TensorRTInferenceGeneric {
protected:
    Logger m_Logger;
    nvinfer1::ICudaEngine* m_Engine = nullptr;
    int m_InputBindingIndex;
    uint64_t m_InputSize;
    nvinfer1::IExecutionContext* m_Context;
    std::vector<void*> m_DeviceBuffers;
    cudaStream_t m_CudaStream;
    std::vector<TensorInfo> m_OutputTensors;
    int m_BatchSize = 1;
    const std::string m_InputBlobName;
    int width = 400;
    int height = 208;
public:
    TensorRTInferenceGeneric(std::string input_blob_name, int _width, int _height);

    virtual void doInference(const unsigned char* input, const uint32_t batchSize);

    virtual void doInference(const cv::Mat & input);

    bool verifyEngine();

    void allocateBuffers();

    void init(const std::string & engine_path);
};

}
#endif