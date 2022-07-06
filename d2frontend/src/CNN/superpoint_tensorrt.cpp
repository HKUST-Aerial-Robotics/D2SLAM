#include "d2frontend/CNN/superpoint_tensorrt.h"
#include "d2frontend/d2frontend_params.h"
#include "ATen/Parallel.h"
#include "d2frontend/utils.h"
#include <d2frontend/CNN/superpoint_common.h>

#define USE_PCA
namespace D2FrontEnd {


#ifdef USE_TENSORRT
SuperPointTensorRT::SuperPointTensorRT(std::string engine_path, 
    std::string _pca_comp,
    std::string _pca_mean,
    int _width, int _height, 
    float _thres, int _max_num, 
    bool _enable_perf):
    TensorRTInferenceGeneric("image", _width, _height), thres(_thres), max_num(_max_num), enable_perf(_enable_perf) {
    at::set_num_threads(1);
    TensorInfo outputTensorSemi, outputTensorDesc;
    outputTensorSemi.blobName = "semi";
    outputTensorDesc.blobName = "desc";
    outputTensorSemi.volume = height*width;
    outputTensorDesc.volume = 1*SP_DESC_RAW_LEN*height/8*width/8;
    m_InputSize = height*width;
    m_OutputTensors.push_back(outputTensorSemi);
    m_OutputTensors.push_back(outputTensorDesc);
    std::cout << "Trying to init TRT engine of SuperPointTensorRT: " << engine_path << " size " << _width << " " << _height << std::endl;
    init(engine_path);

    pca_comp_T = load_csv_mat_eigen(_pca_comp).transpose();
    pca_mean = load_csv_vec_eigen(_pca_mean).transpose();

    std::cout << "pca_comp rows " << pca_comp_T.rows() << "cols " << pca_comp_T.cols() << std::endl;
    std::cout << "pca_mean " << pca_mean.size() << std::endl;
}

void SuperPointTensorRT::inference(const cv::Mat & input, std::vector<cv::Point2f> & keypoints, std::vector<float> & local_descriptors) {
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
    if (_input.type() == CV_8U){
        _input.convertTo(_input, CV_32F, 1/255.0);
    }
    ((CNNInferenceGeneric*) this)->doInference(_input);
    if (params->enable_perf_output) {
        std::cout << "Inference Time " << tic.toc();
    }

    TicToc tic1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    
    auto mProb = at::from_blob(m_OutputTensors[0].hostBuffer, {1, 1, height, width}, options);
    auto mDesc = at::from_blob(m_OutputTensors[1].hostBuffer, {1, SP_DESC_RAW_LEN, height/8, width/8}, options);
    cv::Mat Prob = cv::Mat(height, width, CV_32F, m_OutputTensors[0].hostBuffer);
    if (params->enable_perf_output) {
        std::cout << " from_blob " << tic1.toc();
    }

    TicToc tic2;
    getKeyPoints(Prob, thres, keypoints, width, height, max_num);
    if (params->enable_perf_output) {
        std::cout << " getKeyPoints " << tic2.toc();
    }

    computeDescriptors(mProb, mDesc, keypoints, local_descriptors, width, height, pca_comp_T, pca_mean);
    
    if (params->enable_perf_output) {
        std::cout << " getKeyPoints+computeDescriptors " << tic2.toc() << "inference all" << tic.toc() << "features" << keypoints.size() << "desc size" << local_descriptors.size() << std::endl;
        // cv::Mat heat(height, width, CV_32F, 1);
        // memcpy(heat.data, m_OutputTensors[0].hostBuffer, width*height * sizeof(float));
        // heat.convertTo(heat, CV_8U, 10000);
        // cv::resize(heat, heat, cv::Size(), 2, 2);
        // cv::imshow("Heat", heat);
    }
}
#endif
}