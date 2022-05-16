#include "onnx_generic.h"
#include "../utils.h"
#ifdef USE_ONNX
namespace D2FrontEnd {
class MobileNetVLADONNX: public ONNXInferenceGeneric {
protected:
    std::array<float, NETVLAD_DESC_SIZE> results_;
    std::array<int64_t, 2> output_shape_;
    std::array<int64_t, 4> input_shape_;
public:
    const int descriptor_size = 4096;
    MobileNetVLADONNX(std::string engine_path, int _width, int _height, bool _enable_perf = false) : 
            ONNXInferenceGeneric(engine_path, "image:0", "descriptor:0", _width, _height),
            output_shape_{1, NETVLAD_DESC_SIZE},
            input_shape_{1, _height, _width, 1}
    {
        std::cout << "Trying to init TRT engine of MobileNetVLADONNX@" << engine_path << std::endl;
        init(engine_path);
    }

    std::vector<float> inference(const cv::Mat & input) {
        TicToc tic;
        cv::Mat _input;
        if (input.channels() == 3) {
            cv::cvtColor(input, _input, cv::COLOR_BGR2GRAY);
        } else {
            _input = input;
        }
        if (_input.rows != height || _input.cols != width) {
            cv::resize(_input, _input, cv::Size(width, height));
        } 
        _input.convertTo(_input, CV_32F, 1/255.0);
        doInference(_input.data, 1);
        printf("MobileNetVLADONNX::inference() took %f ms\n", tic.toc());
        return std::vector<float>(results_.begin(), results_.end());
    }

    void init(const std::string & engine_path) {
        input_image = new float[width*height];
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
               input_image, width*height, input_shape_.data(), 4);
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
    }
};
}
#endif
