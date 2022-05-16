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
    Ort::Session session_;
    std::string output_name;
    float * input_image = nullptr;
public:
    ONNXInferenceGeneric(std::string engine_path, std::string input_blob_name, std::string output_blob_name, int _width, int _height): 
        CNNInferenceGeneric(input_blob_name, _width, _height), 
        session_(env, engine_path.c_str(), Ort::SessionOptions{nullptr}),
        output_name(output_blob_name) {}

    virtual void doInference(const unsigned char* input, const uint32_t batchSize) override {
        const char* input_names[] = {m_InputBlobName.c_str()};
        const char* output_names[] = {output_name.c_str()};
        memcpy(input_image, input, width*height*sizeof(float));
        session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    }

    void init(const std::string & engine_path) {
        
    }
};
}
#endif
