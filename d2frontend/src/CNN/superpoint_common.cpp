#include <d2frontend/CNN/superpoint_common.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/utils.h>

#include "d2common/utils.hpp"
using D2Common::Utility::TicToc;

namespace D2FrontEnd {
void NMS2(std::vector<cv::Point2f> det, cv::Mat conf,
          std::vector<cv::Point2f>& pts, std::vector<float>& scores, int border,
          int dist_thresh, int img_width, int img_height, int max_num);
void getKeyPoints(const cv::Mat& prob, float threshold, int nms_dist,
                  std::vector<cv::Point2f>& keypoints,
                  std::vector<float>& scores, int width, int height,
                  int max_num) {
  TicToc getkps;
  auto mask = (prob > threshold);
  std::vector<cv::Point> kps;
  cv::findNonZero(mask, kps);
  std::vector<cv::Point2f> keypoints_no_nms;
  for (unsigned int i = 0; i < kps.size(); i++) {
    keypoints_no_nms.push_back(cv::Point2f(kps[i].x, kps[i].y));
  }

  cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
  for (size_t i = 0; i < keypoints_no_nms.size(); i++) {
    int x = keypoints_no_nms[i].x;
    int y = keypoints_no_nms[i].y;
    conf.at<float>(i, 0) = prob.at<float>(y, x);
  }

  int border = 0;
  TicToc ticnms;
  NMS2(keypoints_no_nms, conf, keypoints, scores, border, nms_dist, width,
       height, max_num);
  if (params->enable_perf_output) {
    printf(" NMS %f keypoints_no_nms %ld keypoints %ld/%ld\n", ticnms.toc(),
           keypoints_no_nms.size(), keypoints.size(), max_num);
  }
}

void computeDescriptors(const torch::Tensor& mProb, const torch::Tensor& mDesc,
                        const std::vector<cv::Point2f>& keypoints,
                        std::vector<float>& local_descriptors, int width,
                        int height, const Eigen::MatrixXf& pca_comp_T,
                        const Eigen::RowVectorXf& pca_mean) {
  TicToc tic;
  cv::Mat kpt_mat(keypoints.size(), 2, CV_32F);  // [n_keypoints, 2]  (y, x)
  for (size_t i = 0; i < keypoints.size(); i++) {
    kpt_mat.at<float>(i, 0) = (float)keypoints[i].y;
    kpt_mat.at<float>(i, 1) = (float)keypoints[i].x;
  }

  auto fkpts =
      at::from_blob(kpt_mat.data, {keypoints.size(), 2}, torch::kFloat);

  auto grid = torch::zeros({1, 1, fkpts.size(0), 2});  // [1, 1, n_keypoints, 2]
  grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / width - 1;   // x
  grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / height - 1;  // y

  // mDesc.to(torch::kCUDA);
  // grid.to(torch::kCUDA);
  auto desc = torch::grid_sampler(mDesc, grid, 0, 0, 0); // [1,256,w/8,h/8] [1,1,n_keypoints,2] [1,256,1,n_keypoints]
  desc = desc.squeeze(0).squeeze(1);

  // normalize to 1
  auto dn = torch::norm(desc, 2, 1);
  desc = desc.div(torch::unsqueeze(dn, 1));

  desc = desc.transpose(0, 1).contiguous();
  desc = desc.to(torch::kCPU);
  Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      _desc(desc.data<float>(), desc.size(0), desc.size(1)); //[256, n_keypoints]
  if (pca_comp_T.size() > 0) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        _desc_new = (_desc.rowwise() - pca_mean) * pca_comp_T;
    // Perform row wise normalization
    for (unsigned int i = 0; i < _desc_new.rows(); i++) {
      _desc_new.row(i) /= _desc_new.row(i).norm();
    }
    local_descriptors = std::vector<float>(
        _desc_new.data(),
        _desc_new.data() + _desc_new.cols() * _desc_new.rows());
  } else {
    for (unsigned int i = 0; i < _desc.rows(); i++) {
      _desc.row(i) /= _desc.row(i).norm();
    }
    local_descriptors = std::vector<float>(
        _desc.data(), _desc.data() + _desc.cols() * _desc.rows());
  }
  if (params->enable_perf_output) {
    std::cout << " computeDescriptors full " << tic.toc() << std::endl;
  }
}

bool pt_conf_comp(std::pair<cv::Point2f, double> i1,
                  std::pair<cv::Point2f, double> i2) {
  return (i1.second > i2.second);
}

// NMS code is modified from https://github.com/KinglittleQ/SuperPoint_SLAM
void NMS2(std::vector<cv::Point2f> det, cv::Mat conf,
          std::vector<cv::Point2f>& pts, std::vector<float>& scores, int border,
          int dist_thresh, int img_width, int img_height, int max_num) {
  std::vector<cv::Point2f> pts_raw = det;

  std::vector<std::pair<cv::Point2f, double>> pts_conf_vec;

  cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
  cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

  cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

  grid.setTo(0);
  inds.setTo(0);
  confidence.setTo(0);

  for (unsigned int i = 0; i < pts_raw.size(); i++) {
    int uu = (int)pts_raw[i].x;
    int vv = (int)pts_raw[i].y;

    grid.at<char>(vv, uu) = 1;
    inds.at<unsigned short>(vv, uu) = i;

    confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
  }

  for (unsigned int i = 0; i < pts_raw.size(); i++) {
    int uu = (int)pts_raw[i].x;
    int vv = (int)pts_raw[i].y;

    if (grid.at<char>(vv, uu) != 1) continue;

    for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
      for (int j = -dist_thresh; j < (dist_thresh + 1); j++) {
        if (j == 0 && k == 0) continue;
        if (uu + j < 0 || uu + j >= img_width || vv + k < 0 ||
            vv + k >= img_height)
          continue;

        if (confidence.at<float>(vv + k, uu + j) <
            confidence.at<float>(vv, uu)) {
          grid.at<char>(vv + k, uu + j) = 0;
        }
      }
    grid.at<char>(vv, uu) = 2;
  }

  size_t valid_cnt = 0;

  for (int v = 0; v < (img_height); v++) {
    for (int u = 0; u < (img_width); u++) {
      if (u >= (img_width - border) || u < border ||
          v >= (img_height - border) || v < border)
        continue;

      if (grid.at<char>(v, u) == 2) {
        int select_ind = (int)inds.at<unsigned short>(v, u);
        float _conf = confidence.at<float>(v, u);
        cv::Point2f p = pts_raw[select_ind];
        pts_conf_vec.push_back(std::make_pair(p, _conf));
        valid_cnt++;
      }
    }
  }

  std::sort(pts_conf_vec.begin(), pts_conf_vec.end(), pt_conf_comp);
  for (unsigned int i = 0; i < max_num && i < pts_conf_vec.size(); i++) {
    pts.push_back(pts_conf_vec[i].first);
    scores.push_back(pts_conf_vec[i].second);
  }
}
}  // namespace D2FrontEnd