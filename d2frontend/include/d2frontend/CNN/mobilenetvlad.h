#include <string>
#include <memory>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "d2common/utils.hpp"
#include "tensorrt_utils/buffers.h"

#define NETVLAD_DESC_RAW_SIZE 4096
namespace D2FrontEnd {
using D2Common::Utility::TicToc;
class MobileNetVLAD{
public:
    struct MobileNetVLADConfig{
        int32_t width;
        int32_t height;
        std::string input_tensor_name;
        std::string output_tensor_name;
        int32_t 
    };

    explicit MobileNetVLAD(const MobileNetVLADConfig &netvald_config){

    }
    bool build();
    bool infer(const cv::Mat &image, std::vector<float> &descriptor);
    void saveEngine();
    bool deserializeEngine();
    
protected:
    MobileNetVLADConfig netvald_config_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims output_dims_{};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    Eigen::MatrixXf pca_comp_T_;
    Eigen::RowVectorXf pca_mean_;
};
}// namespace D2FrontEnd
