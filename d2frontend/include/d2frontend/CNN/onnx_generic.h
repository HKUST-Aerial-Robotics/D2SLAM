#pragma once
#include "CNN_generic.h"

#include <onnxruntime_cxx_api.h>
namespace D2FrontEnd {
class ONNXInferenceGeneric: public CNNInferenceGeneric {
protected:
    Ort::Value input_tensor_{nullptr};
    Ort::Value output_tensor_{nullptr};
    Ort::Env env;
    Ort::Session * session_ = nullptr;
    std::string output_name;
    float * input_image = nullptr;
    char engine_folder [256] = {0};
    char int8_calib_table_name_c [256] = {0};
public:
    ONNXInferenceGeneric(std::string engine_path, std::string input_blob_name, std::string output_blob_name, int _width, int _height,
            bool use_tensorrt, bool use_fp16, bool use_int8, std::string int8_calib_table_name = ""):
        CNNInferenceGeneric(input_blob_name, _width, _height), output_name(output_blob_name) {
        init(engine_path, use_tensorrt, use_fp16, use_int8, int8_calib_table_name) ;
    }

    virtual void doInference(const unsigned char* input, const uint32_t batchSize) override {
        const char* input_names[] = {m_InputBlobName.c_str()};
        const char* output_names[] = {output_name.c_str()};
        memcpy(input_image, input, width*height*sizeof(float));
        session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    }

    void init(const std::string & engine_path, bool onnx_with_tensorrt, bool enable_fp16, bool enable_int8, std::string int8_calib_table_name = "") {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        if (onnx_with_tensorrt) {
            int pn = engine_path.find_last_of('/');
            std::string configPath = engine_path.substr(0, pn);
            memcpy(engine_folder, configPath.c_str(), configPath.size());
            memcpy(int8_calib_table_name_c, int8_calib_table_name.c_str(), int8_calib_table_name.size());
            OrtTensorRTProviderOptions tensorrt_options{};
            tensorrt_options.device_id = 0;
            tensorrt_options.has_user_compute_stream = 0;
            tensorrt_options.trt_fp16_enable = enable_fp16;
            tensorrt_options.trt_int8_enable = enable_int8;
            tensorrt_options.trt_max_workspace_size = 1 * 1024 * 1024 * 1024;
            tensorrt_options.trt_engine_cache_enable = 1;
            tensorrt_options.trt_engine_cache_path = engine_folder;
            tensorrt_options.trt_max_partition_iterations = 10;
            tensorrt_options.trt_min_subgraph_size = 5;
            tensorrt_options.trt_int8_use_native_calibration_table = 0;
            tensorrt_options.trt_int8_calibration_table_name = int8_calib_table_name_c;
            tensorrt_options.trt_dump_subgraphs = 0; 
            session_options.AppendExecutionProvider_TensorRT(tensorrt_options);
            printf("ONNX will use TensorRT for inference INT8 %d FP16 %d cache path %s\n", enable_int8, enable_fp16, engine_folder);
        }

        OrtCUDAProviderOptions options;
        options.device_id = 0;
        options.arena_extend_strategy = 0;
        options.gpu_mem_limit = 1 * 1024 * 1024 * 1024;
        options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(options);

        session_ = new Ort::Session(env, engine_path.c_str(), session_options);
    }
};
}
