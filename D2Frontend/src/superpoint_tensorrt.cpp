#include "swarm_loop/superpoint_tensorrt.h"
#include "swarm_loop/loop_defines.h"
#include "ATen/Parallel.h"
#include "swarm_loop/utils.h"
#include "swarm_msgs/swarm_types.hpp"

#define USE_PCA
using namespace Swarm;
//NMS code is modified from https://github.com/KinglittleQ/SuperPoint_SLAM
void NMS2(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts,
            int border, int dist_thresh, int img_width, int img_height, int max_num);

#define MAXBUFSIZE 100000
Eigen::MatrixXf load_csv_mat_eigen(std::string csv) {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(csv);
    std::string line;

    while (getline(infile, line))
    {
        int temp_cols = 0;
        std::stringstream          lineStream(line);
        std::string                cell;

        while (std::getline(lineStream, cell, ','))
        {
            buff[rows * cols + temp_cols] = std::stod(cell);
            temp_cols ++;
        }

        rows ++;
        if (cols > 0) {
            assert(cols == temp_cols && "Matrix must have same cols on each rows!");
        } else {
            cols = temp_cols;
        }
    }

    infile.close();

    Eigen::MatrixXf result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
}

Eigen::VectorXf load_csv_vec_eigen(std::string csv) {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(csv);
    while (! infile.eof())
    {
        std::string line;
        getline(infile, line);

        int temp_cols = 0;
        std::stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::VectorXf result(rows,cols);
    for (int i = 0; i < rows; i++)
            result(i) = buff[ i ];

    return result;
}

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
    std::cout << "Trying to init TRT engine of SuperPointTensorRT" << engine_path << std::endl;
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
    if (input.rows != height || input.cols != width) {
        cv::resize(input, _input, cv::Size(width, height));
        _input.convertTo(_input, CV_32F, 1/255.0);
    } else {
        input.convertTo(_input, CV_32F, 1/255.0);
    }
    doInference(_input);
    if (enable_perf) {
        std::cout << "Inference Time " << tic.toc();
    }

    TicToc tic1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    
    
    
    auto mProb = at::from_blob(m_OutputTensors[0].hostBuffer, {1, 1, height, width}, options);
    auto mDesc = at::from_blob(m_OutputTensors[1].hostBuffer, {1, SP_DESC_RAW_LEN, height/8, width/8}, options);
    cv::Mat Prob = cv::Mat(height, width, CV_32F, m_OutputTensors[0].hostBuffer);
    if (enable_perf) {
        std::cout << " from_blob " << tic1.toc();
    }

    TicToc tic2;
    getKeyPoints(Prob, thres, keypoints);
    if (enable_perf) {
        std::cout << " getKeyPoints " << tic2.toc();
    }

    computeDescriptors(mProb, mDesc, keypoints, local_descriptors);
    
    if (enable_perf) {
        std::cout << " getKeyPoints+computeDescriptors " << tic2.toc() << "inference all" << tic.toc() << "features" << keypoints.size() << "desc size" << local_descriptors.size() << std::endl;
        // cv::Mat heat(height, width, CV_32F, 1);
        // memcpy(heat.data, m_OutputTensors[0].hostBuffer, width*height * sizeof(float));
        // heat.convertTo(heat, CV_8U, 10000);
        // cv::resize(heat, heat, cv::Size(), 2, 2);
        // cv::imshow("Heat", heat);
    }
}

void SuperPointTensorRT::getKeyPoints(const cv::Mat & prob, float threshold, std::vector<cv::Point2f> &keypoints)
{
    TicToc getkps;
    auto mask = (prob > threshold);
    std::vector<cv::Point> kps;
    cv::findNonZero(mask, kps);
    std::vector<cv::Point2f> keypoints_no_nms;
    for (int i = 0; i < kps.size(); i++) {
        keypoints_no_nms.push_back(cv::Point2f(kps[i].x, kps[i].y));
    }

    cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
    for (size_t i = 0; i < keypoints_no_nms.size(); i++) {
        int x = keypoints_no_nms[i].x;
        int y = keypoints_no_nms[i].y;
        conf.at<float>(i, 0) = prob.at<float>(y, x);
    }

    int border = 0;
    int dist_thresh = 4;
    TicToc ticnms;
    NMS2(keypoints_no_nms, conf, keypoints, border, dist_thresh, width, height, max_num);
    if (enable_perf) {
        printf(" NMS %f keypoints_no_nms %ld keypoints %ld\n", ticnms.toc(), keypoints_no_nms.size(), keypoints.size());
    }
}


void SuperPointTensorRT::computeDescriptors(const torch::Tensor & mProb, const torch::Tensor & mDesc, const std::vector<cv::Point2f> &keypoints, std::vector<float> & local_descriptors) {
    TicToc tic;
    cv::Mat kpt_mat(keypoints.size(), 2, CV_32F);  // [n_keypoints, 2]  (y, x)
    for (size_t i = 0; i < keypoints.size(); i++) {
        kpt_mat.at<float>(i, 0) = (float)keypoints[i].y;
        kpt_mat.at<float>(i, 1) = (float)keypoints[i].x;
    }


    auto fkpts = torch::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);

    auto grid = torch::zeros({1, 1, fkpts.size(0), 2});  // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / width - 1;  // x
    grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / height - 1;  // y

    // mDesc.to(torch::kCUDA);
    // grid.to(torch::kCUDA);
    auto desc = torch::grid_sampler(mDesc, grid, 0, 0, 0);

    desc = desc.squeeze(0).squeeze(1);

    // normalize to 1
    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    desc = desc.transpose(0, 1).contiguous();
    desc = desc.to(torch::kCPU);
    Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> _desc(desc.data<float>(), desc.size(0), desc.size(1));
#ifdef USE_PCA
    Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> _desc_new = (_desc.rowwise() - pca_mean) *pca_comp_T;
    local_descriptors = std::vector<float>(_desc_new.data(), _desc_new.data()+_desc_new.cols()*_desc_new.rows());
#else
    local_descriptors = std::vector<float>(_desc.data(), _desc.data()+_desc.cols()*_desc.rows());
#endif

    if (enable_perf) {
        std::cout << " computeDescriptors full " << tic.toc() << std::endl;
    }
}

bool pt_conf_comp(std::pair<cv::Point2f, double> i1, std::pair<cv::Point2f, double> i2)
{
    return (i1.second > i2.second);
}

void NMS2(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts,
            int border, int dist_thresh, int img_width, int img_height, int max_num)
{

    std::vector<cv::Point2f> pts_raw = det;

    std::vector<std::pair<cv::Point2f, double>> pts_conf_vec;

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);

    for (unsigned int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }

    for (int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++)
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                if(j==0 && k==0) continue;

                if ( confidence.at<float>(vv + k, uu + j) < confidence.at<float>(vv, uu) ) {
                    grid.at<char>(vv + k, uu + j) = 0;
                }
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;

    for (int v = 0; v < (img_height); v++){
        for (int u = 0; u < (img_width); u++)
        {
            if (u>= (img_width - border) || u < border || v >= (img_height - border) || v < border)
            continue;

            if (grid.at<char>(v,u) == 2)
            {
                int select_ind = (int) inds.at<unsigned short>(v, u);
                float _conf = confidence.at<float> (v, u);
                cv::Point2f p = pts_raw[select_ind];
                pts_conf_vec.push_back(std::make_pair(p, _conf));
                valid_cnt++;
            }
        }
    }
    
    std::sort(pts_conf_vec.begin(), pts_conf_vec.end(), pt_conf_comp);
    for (unsigned int i = 0; i < max_num && i < pts_conf_vec.size(); i ++) {
        pts.push_back(pts_conf_vec[i].first);
        // printf("conf:%f\n", pts_conf_vec[i].second);
    }

}