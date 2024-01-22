#include <d2frontend/CNN/superpoint_common.h>
#include <d2frontend/utils.h>
#include <d2frontend/d2frontend_params.h>
#include <spdlog/spdlog.h>
#include "d2common/utils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/types.hpp>

using D2Common::Utility::TicToc;
#define evaluation 0

namespace D2FrontEnd {
void NMS2(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts, std::vector<float>& scores,
        int border, int dist_thresh, int img_width, int img_height, int max_num);
void FastNMS2(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts, 
            std::vector<float>& scores, int border, int dist_thresh, int img_width, int img_height, int max_num);

void getKeyPoints(const cv::Mat & prob, float threshold, int nms_dist, std::vector<cv::Point2f> &keypoints, std::vector<float>& scores, int width, int height, int max_num ,  int32_t cam_id)
{
    TicToc getkps;
    #if 0
    static int count = 0;
    std::string prob_path = "/root/swarm_ws/src/D2SLAM/super_point_evaluation/prob/";
    std::string file_name = prob_path + std::to_string(count) + ".png";
    // cv::imwrite(file_name, prob);
    cv::imshow("prob", prob);
    count++;
    #endif

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

    int border = 10;
    TicToc ticnms;
    //TODO This funciton delay the whole frontend  input: rough keypoints and confidence map
    // NMS2(keypoints_no_nms, conf, keypoints, scores, border, nms_dist, width, height, max_num);
    FastNMS2(keypoints_no_nms, conf, keypoints, scores, border, nms_dist, width, height, max_num);
    if (params->enable_perf_output) {
        printf("cam_id %d NMS %f keypoints_no_nms %ld keypoints %ld/%ld\n",cam_id, ticnms.toc(), keypoints_no_nms.size(), keypoints.size(), max_num);
    }
}


void computeDescriptors(const torch::Tensor & mProb, const torch::Tensor & mDesc, 
        const std::vector<cv::Point2f> &keypoints, 
        std::vector<float> & local_descriptors, int width, int height, 
        const Eigen::MatrixXf & pca_comp_T, const Eigen::RowVectorXf & pca_mean) {
    TicToc tic;
    cv::Mat kpt_mat(keypoints.size(), 2, CV_32F);  // [n_keypoints, 2]  (y, x)
    for (size_t i = 0; i < keypoints.size(); i++) {
        kpt_mat.at<float>(i, 0) = (float)keypoints[i].y;
        kpt_mat.at<float>(i, 1) = (float)keypoints[i].x;
    }


    auto fkpts = at::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);

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
    if (pca_comp_T.size() > 0) {
        Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> _desc_new = (_desc.rowwise() - pca_mean) *pca_comp_T;
        //Perform row wise normalization
        for (int i = 0; i < _desc_new.rows(); i++) {
            _desc_new.row(i) /= _desc_new.row(i).norm();
        }
        local_descriptors = std::vector<float>(_desc_new.data(), _desc_new.data()+_desc_new.cols()*_desc_new.rows());
    } else {
        for (int i = 0; i < _desc.rows(); i++) {
            _desc.row(i) /= _desc.row(i).norm();
        }
        local_descriptors = std::vector<float>(_desc.data(), _desc.data()+_desc.cols()*_desc.rows());
    }
    if (params->enable_perf_output) {
        std::cout << " computeDescriptors full " << tic.toc() << std::endl;
    }
}

bool pt_conf_comp(std::pair<cv::Point2f, double> i1, std::pair<cv::Point2f, double> i2)
{
    return (i1.second > i2.second);
}

//NMS code is modified from https://github.com/KinglittleQ/SuperPoint_SLAM

void FastNMS2(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts, 
            std::vector<float>& scores, int border, int dist_thresh, int img_width, int img_height, int max_num)
{
    struct PointConf{
        cv::Point2f pt;
        float conf;
        bool operator()(const PointConf &a, const PointConf &b) const
        {
            return a.conf > b.conf;
        }
    };
    std::vector<PointConf> raw_points;
    std::vector<cv::Point2f> pts_raw = det;
    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);
    std::vector<std::pair<cv::Point2f, double>> pts_conf_vec;
  
    D2Common::Utility::TicToc tic;

    raw_points.resize(pts_raw.size());
    for (unsigned int i = 0; i < pts_raw.size(); i++)
    {   
        raw_points[i].pt = pts_raw[i];
        raw_points[i].conf = conf.at<float>(i, 0);
    }

    std::sort(raw_points.begin(), raw_points.end(), PointConf());
    size_t valid_cnt = 0;

    tic.tic();

    std::vector <PointConf> top_points_inner;
    for (int i = 0; i < raw_points.size(); i++)
    {   
        int uu = (int) raw_points[i].pt.x;
        int vv = (int) raw_points[i].pt.y;

        if (uu - border < 0 || uu + border >= img_width || vv - border < 0 || vv + border >= img_height){
            continue;
        }
        top_points_inner .push_back(raw_points[i]);
    }
        
    tic.tic();
    for (unsigned int i = 0; i < max_num && i < top_points_inner.size(); i ++) {
        pts.push_back(top_points_inner[i].pt);
        scores.push_back(top_points_inner[i].conf);
    }

    #if evaluation
    static int32_t image_seq = 0;
    std::string image_path = "/root/swarm_ws/src/D2SLAM/super_point_evaluation/image/";
    std::string image_name = image_path + std::to_string(image_seq) + ".png";
    cv::Mat localtion_map = cv::Mat(cv::Size(img_width, img_height), CV_8U);
    localtion_map.setTo(0);
    for (int i = 0; i < pts.size(); i++)
    {
        localtion_map.at<char>(pts[i].x, pts[i].y) = 255;
    }
    cv::imwrite(image_name, localtion_map);
    cv::imshow("localtion_map", localtion_map);
    cv::waitKey(0);
    #endif

    // spdlog::info("NMS2 sort {} ms", tic.toc());
}

//NMS low

// NMS Fast version
void NMS2(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts, 
            std::vector<float>& scores, int border, int dist_thresh, int img_width, int img_height, int max_num)
{

    std::vector<cv::Point2f> pts_raw = det;

    std::vector<std::pair<cv::Point2f, double>> pts_conf_vec;

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);
    cv::setNumThreads(0);
    
    D2Common::Utility::TicToc tic;


    for (unsigned int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }

    tic.tic();

    //There is a 10^6 loop here, need speed up
    int count = 0;
    for (int i = 0; i < pts_raw.size(); i++)
    {   

        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        if (grid.at<char>(vv, uu) != 1){
            continue;
        }

        if (uu - dist_thresh < 0 || uu + dist_thresh >= img_width || vv - dist_thresh < 0 || vv + dist_thresh >= img_height){
            continue;
        }

        for(int k = -dist_thresh; k < (dist_thresh+1); k++){
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                count++;
                if(j==0 && k==0) continue;
                // if (uu+j < 0 || uu+j >= img_width || vv+k < 0 || vv+k >= img_height) continue;
                if ( confidence.at<float>(vv + k, uu + j) < confidence.at<float>(vv, uu) ) {
                    grid.at<char>(vv + k, uu + j) = 0;
                }
            }
        }
        grid.at<char>(vv, uu) = 2;
    }
    // spdlog::info("NMS2 NMS calculate {} ms count:{} \n", tic.toc(),count);

    size_t valid_cnt = 0;

    tic.tic();
    for (int v = 0; v < (img_height); v++){
        if (v < border || v >= (img_height - border)){
            continue;
        }
        for (int u = 0; u < (img_width); u++)
        {
            if (u < border || u >= (img_width - border)){
                continue;
            }

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
    
    tic.tic();
    std::sort(pts_conf_vec.begin(), pts_conf_vec.end(), pt_conf_comp);
    for (unsigned int i = 0; i < max_num && i < pts_conf_vec.size(); i ++) {
        pts.push_back(pts_conf_vec[i].first);
        scores.push_back(pts_conf_vec[i].second);
    }
    // spdlog::info("NMS2 sort {} ms", tic.toc());
}

//NMS low

// NMS Fast version
// void NMS2(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts, 
//             std::vector<float>& scores, int border, int dist_thresh, int img_width, int img_height, int max_num)
// {

//     std::vector<cv::Point2f> pts_raw = det;

//     std::vector<std::pair<cv::Point2f, double>> pts_conf_vec;

//     cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
//     cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

//     cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

//     grid.setTo(0);
//     inds.setTo(0);
//     confidence.setTo(0);
//     cv::setNumThreads(0);
    
//     D2Common::Utility::TicToc tic;


//     for (unsigned int i = 0; i < pts_raw.size(); i++)
//     {   
//         int uu = (int) pts_raw[i].x;
//         int vv = (int) pts_raw[i].y;

//         grid.at<char>(vv, uu) = 1;
//         inds.at<unsigned short>(vv, uu) = i;

//         confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
//     }

//     tic.tic();

//     //There is a 10^6 loop here, need speed up
//     int count = 0;
//     for (int i = 0; i < pts_raw.size(); i++)
//     {   

//         int uu = (int) pts_raw[i].x;
//         int vv = (int) pts_raw[i].y;

//         if (grid.at<char>(vv, uu) != 1){
//             continue;
//         }

//         if (uu - dist_thresh < 0 || uu + dist_thresh >= img_width || vv - dist_thresh < 0 || vv + dist_thresh >= img_height){
//             continue;
//         }

//         for(int k = -dist_thresh; k < (dist_thresh+1); k++){
//             for(int j = -dist_thresh; j < (dist_thresh+1); j++)
//             {
//                 count++;
//                 if(j==0 && k==0) continue;
//                 // if (uu+j < 0 || uu+j >= img_width || vv+k < 0 || vv+k >= img_height) continue;
//                 if ( confidence.at<float>(vv + k, uu + j) < confidence.at<float>(vv, uu) ) {
//                     grid.at<char>(vv + k, uu + j) = 0;
//                 }
//             }
//         }
//         grid.at<char>(vv, uu) = 2;
//     }
//     // spdlog::info("NMS2 NMS calculate {} ms count:{} \n", tic.toc(),count);

//     size_t valid_cnt = 0;

//     tic.tic();
//     for (int v = 0; v < (img_height); v++){
//         if (v < border || v >= (img_height - border)){
//             continue;
//         }
//         for (int u = 0; u < (img_width); u++)
//         {
//             if (u < border || u >= (img_width - border)){
//                 continue;
//             }

//             if (grid.at<char>(v,u) == 2)
//             {
//                 int select_ind = (int) inds.at<unsigned short>(v, u);
//                 float _conf = confidence.at<float> (v, u);
//                 cv::Point2f p = pts_raw[select_ind];
//                 pts_conf_vec.push_back(std::make_pair(p, _conf));
//                 valid_cnt++;
//             }
//         }
//     }
    
//     tic.tic();
//     std::sort(pts_conf_vec.begin(), pts_conf_vec.end(), pt_conf_comp);
//     for (unsigned int i = 0; i < max_num && i < pts_conf_vec.size(); i ++) {
//         pts.push_back(pts_conf_vec[i].first);
//         scores.push_back(pts_conf_vec[i].second);
//     }
//     // spdlog::info("NMS2 sort {} ms", tic.toc());
// }

#if 0
void NMS2(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts, 
            std::vector<float>& scores, int border, int dist_thresh, int img_width, int img_height, int max_num)
{

    std::vector<cv::Point2f> pts_raw = det;

    std::vector<std::pair<cv::Point2f, double>> pts_conf_vec;

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);
    cv::setNumThreads(0);
    
    D2Common::Utility::TicToc tic;


    for (unsigned int i = 0; i < pts_raw.size(); i++)
    {   
        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }
    spdlog::info("NMS2 init {} ms", tic.toc());  

    tic.tic();

    //There is a 10^6 loop here, need speed up
    int count = 0;
    for (int i = 0; i < pts_raw.size(); i++)
    {   

        int uu = (int) pts_raw[i].x;
        int vv = (int) pts_raw[i].y;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for(int k = -dist_thresh; k < (dist_thresh+1); k++){
            for(int j = -dist_thresh; j < (dist_thresh+1); j++)
            {
                count++;
                if(j==0 && k==0) continue;
                if (uu+j < 0 || uu+j >= img_width || vv+k < 0 || vv+k >= img_height) continue;

                if ( confidence.at<float>(vv + k, uu + j) < confidence.at<float>(vv, uu) ) {
                    grid.at<char>(vv + k, uu + j) = 0;
                }
            }
        }
        grid.at<char>(vv, uu) = 2;
    }
    spdlog::info("NMS2 loop {} ms count:{} \n", tic.toc(),count);

    size_t valid_cnt = 0;

    tic.tic();
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
    spdlog::info("NMS2 loop2 {} ms", tic.toc());
    
    tic.tic();
    std::sort(pts_conf_vec.begin(), pts_conf_vec.end(), pt_conf_comp);
    for (unsigned int i = 0; i < max_num && i < pts_conf_vec.size(); i ++) {
        pts.push_back(pts_conf_vec[i].first);
        scores.push_back(pts_conf_vec[i].second);
    }
    spdlog::info("NMS2 sort {} ms", tic.toc());
}
#endif

//CUDA NMS
// void NMSCUDA(std::vector<cv::Point2f> det, cv::Mat conf, std::vector<cv::Point2f>& pts, 
//             std::vector<float>& scores, int border, int dist_thresh, int img_width, int img_height, int max_num){
//     cv::Rect2d
    
// }


}//namespace D2FrontEnd