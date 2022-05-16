#pragma once
#include "CNN_generic.h"
#ifdef USE_ONNX
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
namespace D2FrontEnd {
class ONNXInferenceGeneric: public CNNInferenceGeneric {
protected:
    Ort::Value input_tensor_{nullptr};
    Ort::Value output_tensor_{nullptr};
    Ort::Env env;
    Ort::Session * session_ = nullptr;
    std::string output_name;
    float * input_image = nullptr;
public:
    ONNXInferenceGeneric(std::string engine_path, std::string input_blob_name, std::string output_blob_name, int _width, int _height): 
        CNNInferenceGeneric(input_blob_name, _width, _height), 
        output_name(output_blob_name) {
            init(engine_path);
        }

    virtual void doInference(const unsigned char* input, const uint32_t batchSize) override {
        const char* input_names[] = {m_InputBlobName.c_str()};
        const char* output_names[] = {output_name.c_str()};
        memcpy(input_image, input, width*height*sizeof(float));
        session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    }

    void init(const std::string & engine_path) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        OrtCUDAProviderOptions options;
        options.device_id = 0;
        options.arena_extend_strategy = 0;
        options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
        options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(options);

        session_ = new Ort::Session(env, engine_path.c_str(), session_options);
    }
};
}
#endif
