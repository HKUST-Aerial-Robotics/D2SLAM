#pragma once
#include <d2frontend/CNN/onnx_generic.h>
#include <d2common/utils.hpp>
#include <vector>

using D2Common::Utility::TicToc;

namespace D2QuadCamDepthEst {
class HitnetONNX: public D2FrontEnd::ONNXInferenceGeneric {
public:
    HitnetONNX(std::string engine_path, int net_width, int net_height, bool use_tensorrt = true, 
        bool use_fp16 = true, bool use_int8 = false, bool use_rgb = false, int32_t number_stereo_pairs = 1): 
            ONNXInferenceGeneric(engine_path, "input", "reference_output_disparity", net_width, net_height, use_tensorrt, use_fp16, use_int8),
        output_shape_{1, net_width, net_height, 1}, number_stereo_pairs_{number_stereo_pairs}{
        std::cout << "Trying to init HitnetONNX@" << engine_path << std::endl;

        // this->input_image_mat = cv::Mat(height*2, width, CV_32F);//create a 4 channel images input
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        if(use_rgb){
            ROS_INFO("[ONNX HITNET] USE RGB mode assume input image have three channel\n");
            this->input_shape_ = {this->number_stereo_pairs_, 3, height*2, width};
        } else {
            ROS_INFO("[ONNX HITNET] USE GRAY mode assume input image have one  float grey  channel\n");
            this->input_shape_ = {this->number_stereo_pairs_, 1, height*2, width};
        }
        for(int32_t i = 0 ;i<this->number_stereo_pairs_ ; i ++){
            //construct input image mat
            cv::Mat input_image_mat;
            if(use_rgb){
                input_image_mat = cv::Mat(height*2, width, CV_32FC3);
                this->input_image_mat_vec.push_back(input_image_mat);
            } else {
                input_image_mat = cv::Mat(height*2, width, CV_32F);
                this->input_image_mat_vec.push_back(input_image_mat);
            }
            //create input_tensor
            this->input_tensors_.push_back(Ort::Value::CreateTensor<float>(memory_info,
                input_image_mat.ptr<float>() , 2*width*height, 
                this->input_shape_.data(), this->input_shape_.size()));
            //create output_tensor
            // float* data_ptr = new float[this->width * this->height];
            // if(data_ptr == nullptr){
            //     ROS_ERROR("[ONNX HITNET] Failed to allocate memory for output tensor\n");
            //     exit(-1);
            // }
            // this->output_data_ptr_vec.push_back(data_ptr);
            // this->output_tensors_.push_back(Ort::Value::CreateTensor<float>(memory_info,
            //     data_ptr, this->width* this->height, this->output_shape_.data(), this->output_shape_.size()));

        }
        // this->output_tensor_ = Ort::Value::CreateTensor<float>(memory_info,
        //     p_results_, width*height, output_shape_.data(), output_shape_.size());
        printf("[ONNX HITNET] initialized\n");
    }
    
    ~HitnetONNX(){
        if(this->output_data_ptr_vec.size()>0){
            for(int i = 0 ; i < this->output_data_ptr_vec.size(); i++){
                free(this->output_data_ptr_vec[i]);
            }
        }
        printf("release molloc\n");
    }

    virtual void doInference() {
        const char* input_names[] = {m_InputBlobName.c_str()};
        const char* output_names[] = {output_name.c_str()};
        printf("[ONNX HITNET] run intference\n");
        std::vector<Ort::Value> ouput_tensors_local;

        // session_->Run(Ort::RunOptions{nullptr}, input_names, &this->input_tensors_, 1, output_names, &this->output_tensors_, 1);
        this->session_->Run(Ort::RunOptions{nullptr}, input_names, this->input_tensors_.data(), 
            this->input_tensors_.size(), output_names ,ouput_tensors_local.data(), 4);
        printf("get ouput tensor\n");
    }

    cv::Mat inference(const cv::Mat & input_left, const cv::Mat & input_right) {
        TicToc tic;
        cv::Mat _input_left, _input_right;
        if (input_left.channels() == 3) {
            cv::cvtColor(input_left, _input_left, cv::COLOR_BGR2GRAY);
            cv::cvtColor(input_right, _input_right, cv::COLOR_BGR2GRAY);
        } else {
            _input_left = input_left;
            _input_right = input_right;
        }
        if (_input_left.rows != this->height || _input_left.cols != this->width) {
            cv::resize(_input_left, _input_left, cv::Size(this->width, this->height));
            cv::resize(_input_right, _input_right, cv::Size(this->width, this->height));
        } 
        cv::Mat input;
        cv::vconcat(_input_left, _input_right, input);
        input.convertTo(input_image_mat, CV_32F, 1.0/255.0);
        // doInference();
        printf("HitnetONNX::inference() took %f ms\n", tic.toc());
        return cv::Mat(this->height, this->width, CV_32F, this->p_results_);
    }

protected:
    float * p_results_;
    std::array<int64_t, 4> output_shape_;
    std::array<int64_t, 4> input_shape_;
    cv::Mat input_image_mat;
    int32_t number_stereo_pairs_;
    std::vector<Ort::Value> input_tensors_;
    std::vector<Ort::Value> output_tensors_;

private:
    std::vector<cv::Mat> input_image_mat_vec;
    std::vector<float*> output_data_ptr_vec;
};
}