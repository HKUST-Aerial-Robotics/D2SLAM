#pragma once
#include "tensorrt_generic.h"
#ifdef USE_TENSORRT

#include <Eigen/Dense>

namespace D2FrontEnd {
#ifdef USE_TENSORRT
class SuperPointTensorRT: public TensorRTInferenceGeneric {
    Eigen::MatrixXf pca_comp_T;
    Eigen::RowVectorXf pca_mean;

public:
    double thres = 0.015;
    bool enable_perf;
    int max_num = 200;
    SuperPointTensorRT(std::string engine_path, 
        std::string _pca_comp,
        std::string _pca_mean,
        int _width, int _height, float _thres = 0.015, int _max_num = 200, bool _enable_perf = false);
    void inference(const cv::Mat & input, std::vector<cv::Point2f> & keypoints, std::vector<float> & local_descriptors);
};
#endif

}
#endif