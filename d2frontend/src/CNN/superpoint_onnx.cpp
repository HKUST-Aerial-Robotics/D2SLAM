#include <d2frontend/CNN/superpoint_onnx.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/CNN/superpoint_common.h>
#include <d2frontend/utils.h>
#include "ATen/Parallel.h"
#include "d2common/utils.hpp"
using D2Common::Utility::TicToc;

#ifdef USE_ONNX
namespace D2FrontEnd {
SuperPointONNX::SuperPointONNX(std::string engine_path, 
    std::string _pca_comp,
    std::string _pca_mean,
    int _width, int _height, 
    float _thres, int _max_num, 
    bool _enable_perf):
        ONNXInferenceGeneric(engine_path, "image", "semi", _width, _height),
        output_shape_semi_{1, _height, _width},
        output_shape_desc_{1, SP_DESC_RAW_LEN, _height/8, _width/8},
        input_shape_{1, 1, _height, _width},
        max_num(_max_num) {
    at::set_num_threads(1);
    std::cout << "Init SuperPointONNX: " << engine_path << " size " << _width << " " << _height << std::endl;

    input_image = new float[_width*_height];
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image, width*height, input_shape_.data(), 4);

    results_desc_ = new float[1*SP_DESC_RAW_LEN*height/8*width/8];
    results_semi_ = new float[width*height];
    //semi
    output_tensors_.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
        results_semi_, height*width, output_shape_semi_.data(), output_shape_semi_.size()));
    //desc
    output_tensors_.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
        results_desc_, SP_DESC_RAW_LEN*height/8*width/8, output_shape_desc_.data(), output_shape_desc_.size()));

#ifdef USE_PCA
    pca_comp_T = load_csv_mat_eigen(_pca_comp).transpose();
    pca_mean = load_csv_vec_eigen(_pca_mean).transpose();
#endif
}

void SuperPointONNX::doInference(const unsigned char* input, const uint32_t batchSize) {
    const char* input_names[] = {m_InputBlobName.c_str()};
    const char* output_names_[] = {"semi", "desc"};
    memcpy(input_image, input, width*height*sizeof(float));
    session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names_, output_tensors_.data(), 2);
}

void SuperPointONNX::inference(const cv::Mat & input, std::vector<cv::Point2f> & keypoints, std::vector<float> & local_descriptors, std::vector<float> & scores) {
    TicToc tic;
    cv::Mat _input;
    keypoints.clear();
    local_descriptors.clear();
    assert(input.rows == height && input.cols == width && "Input image must have same size with network");
    if (input.channels() == 3) {
        cv::cvtColor(input, _input, cv::COLOR_BGR2GRAY);
    } else {
        _input = input;
    }
    if (_input.rows != height || _input.cols != width) {
        cv::resize(_input, _input, cv::Size(width, height));
    } 
    _input.convertTo(_input, CV_32F, 1/255.0);
    ((CNNInferenceGeneric*) this)->doInference(_input);
    if (params->enable_perf_output) {
        std::cout << "Inference Time " << tic.toc();
    }

    TicToc tic1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    
    auto mProb = at::from_blob(results_semi_, {1, 1, height, width}, options);
    auto mDesc = at::from_blob(results_desc_, {1, SP_DESC_RAW_LEN, height/8, width/8}, options);
    cv::Mat Prob(height, width, CV_32F, results_semi_);
    if (params->enable_perf_output) {
        std::cout << " from_blob " << tic1.toc();
    }

    TicToc tic2;
    getKeyPoints(Prob, thres, keypoints, scores, width, height, max_num);
    if (params->enable_perf_output) {
        std::cout << " getKeyPoints " << tic2.toc();
    }
    computeDescriptors(mProb, mDesc, keypoints, local_descriptors, width, height, pca_comp_T, pca_mean);
    if (params->enable_perf_output) {
        std::cout << " getKeyPoints+computeDescriptors " << tic2.toc() << "inference all" << tic.toc() << "features" << keypoints.size() << "desc size" << local_descriptors.size() << std::endl;
    }
}
}
#endif