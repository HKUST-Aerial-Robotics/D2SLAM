#pragma once
//Original code from https://github.com/enazoe/yolo-tensorrt
#include <opencv2/opencv.hpp>
namespace D2FrontEnd {
class CNNInferenceGeneric {
protected:
    std::string m_InputBlobName;
    int width = 400;
    int height = 208;
    float * data_buf = nullptr;
public:
    CNNInferenceGeneric(std::string input_blob_name, int _width, int _height):
        m_InputBlobName(input_blob_name), width(_width), height(_height) {}
    virtual void doInference(const cv::Mat & input) {
        if (input.channels() == 1) {
            doInference(input.data, 1);
        } else {
            cv::Mat bgr[3];
            cv::split(input, bgr);
            if (data_buf == nullptr) {
                data_buf = new float[3*input.rows*input.cols];
            }
            memcpy(data_buf, bgr[2].data, input.rows*input.cols*sizeof(float));
            memcpy(data_buf+input.rows*input.cols, bgr[1].data, input.rows*input.cols*sizeof(float));
            memcpy(data_buf+input.rows*input.cols*2, bgr[0].data, input.rows*input.cols*sizeof(float));
            doInference((unsigned char*)data_buf, 1);
        }
    }
    virtual void doInference(const unsigned char* input, const uint32_t batchSize) {}
    virtual void init(const std::string & engine_path) {}
};
}