#pragma once
//Original code from https://github.com/enazoe/yolo-tensorrt
#include <opencv2/opencv.hpp>
#include "CNN_generic.h"

#ifdef USE_TENSORRT
#include "NvInfer.h"
#include <trt_utils.h>
namespace D2FrontEnd {
struct TensorInfo
{
    std::string blobName;
    float* hostBuffer{nullptr};
    uint64_t volume{0};
    int bindingIndex{-1};
};

class TensorRTInferenceGeneric: public CNNInferenceGeneric {
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
public:
    TensorRTInferenceGeneric(std::string input_blob_name, int _width, int _height);

    virtual void doInference(const unsigned char* input, const uint32_t batchSize) override;

    bool verifyEngine();

    void allocateBuffers();

    void init(const std::string & engine_path);
};

}
#endif