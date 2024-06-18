#include <boost/program_options.hpp>

#include "d2frontend/CNN/mobilenetvlad_onnx.h"
#include "d2frontend/CNN/superglue_onnx.h"
#include "d2frontend/CNN/superpoint_onnx.h"
#include "d2frontend/d2frontend_params.h"
#include "d2frontend/utils.h"

using namespace Swarm;
using namespace D2FrontEnd;
D2FrontendParams* D2FrontEnd::params = new D2FrontendParams;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    return -1;
  }
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "superpoint,s", po::value<std::string>()->default_value(""),
      "model path of SuperPoint")("netvlad,n",
                                  po::value<std::string>()->default_value(""),
                                  "model path of NetVLAD")(
      "superglue,g", po::value<std::string>()->default_value(""),
      "model path of SuperGlue")("limage,l",
                                 po::value<std::string>()->default_value(""),
                                 "model path of image0")(
      "rimage,r", po::value<std::string>()->default_value(""),
      "model path of image1")(
      "focal,f", po::value<double>()->default_value(384), "image focal length")(
      "width,w", po::value<int>()->default_value(640), "image width")(
      "height,h", po::value<int>()->default_value(480), "image height")(
      "tensorrt", po::value<bool>()->default_value(1), "use tensorrt")(
      "num-test,t", po::value<int>()->default_value(100), "num of tests");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  // params->enable_perf_output = true;

  int width = vm["width"].as<int>();
  int height = vm["height"].as<int>();
  int num_test = vm["num-test"].as<int>();
  double focal = vm["focal"].as<double>();
  bool use_tensorrt = vm["tensorrt"].as<bool>();
  printf("width: %d, height: %d, num_test: %d, focal: %f, use_tensorrt: %d\n",
         width, height, num_test, focal, use_tensorrt);

  cv::Mat img0 = cv::imread(vm["limage"].as<std::string>());
  cv::Mat img1;
  if (vm["rimage"].as<std::string>() != "") {
    img1 = cv::imread(vm["rimage"].as<std::string>());
  } else {
    img1 = img0.clone();
  }
  std::vector<float> local_desc0, local_desc1, scores0, scores1;
  std::vector<cv::Point2f> kps0, kps1;
  std::vector<cv::Point2f> kps0_norm, kps1_norm;

  cv::Mat img_gray0, img_gray1;
  cv::resize(img0, img_gray0, cv::Size(width, height));
  cv::resize(img1, img_gray1, cv::Size(width, height));
  cv::cvtColor(img_gray0, img_gray0, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img_gray1, img_gray1, cv::COLOR_BGR2GRAY);
  cv::Mat show;

  SuperGlueOnnx sg_onnx(vm["superglue"].as<std::string>());
  MobileNetVLADONNX netvlad_onnx(vm["netvlad"].as<std::string>(), 640, 480,
                                 use_tensorrt);
  SuperPointONNX sp_onnx(vm["superpoint"].as<std::string>(), 20, "", "", 640,
                         480, 0.015, 200, use_tensorrt);
  std::cout << "Finish loading models" << std::endl;

  sp_onnx.inference(img_gray0, kps0, local_desc0, scores0);
  sp_onnx.inference(img_gray1, kps1, local_desc1, scores1);
  std::cout << "Finish inference superpoint features" << kps0.size() << ":"
            << kps1.size() << std::endl;
  Eigen::Map<VectorXf> desc0(local_desc0.data(), local_desc0.size());
  Eigen::Map<VectorXf> desc1(local_desc1.data(), local_desc1.size());
  std::cout << "minmax coeff of superpoint" << desc0.minCoeff() << ":"
            << desc0.maxCoeff() << std::endl;
  std::cout << "minmax coeff of superpoint" << desc1.minCoeff() << ":"
            << desc1.maxCoeff() << std::endl;
  double max0 = desc0.cwiseAbs().maxCoeff();
  double max1 = desc1.cwiseAbs().maxCoeff();
  // Convert to int8 with scale maxCoeff()/127
  VectorXi desc_i0 = (desc0 / max0 * 127).template cast<int>();
  VectorXi desc_i1 = (desc1 / max1 * 127).template cast<int>();

  // Recover float from int8
  desc0 = desc_i0.template cast<float>() / 127 * max0;
  desc1 = desc_i1.template cast<float>() / 127 * max1;

  auto global_desc0 = netvlad_onnx.inference(img_gray0);
  auto global_desc1 = netvlad_onnx.inference(img_gray1);
  VectorXf gdesc0 = VectorXf::Map(global_desc0.data(), global_desc0.size());
  VectorXf gdesc1 = VectorXf::Map(global_desc1.data(), global_desc1.size());
  std::cout << "minmax coeff of netvlad0 " << gdesc0.minCoeff() << ":"
            << gdesc0.maxCoeff() << "norm: " << gdesc0.norm() << std::endl;
  std::cout << "minmax coeff of netvlad1 " << gdesc1.minCoeff() << ":"
            << gdesc1.maxCoeff() << "norm: " << gdesc1.norm() << std::endl;
  for (size_t i = 0; i < kps0.size(); i++) {
    kps0_norm.emplace_back(cv::Point2f((kps0[i].x - width / 2) / focal,
                                       (kps0[i].y - height / 2) / focal));
  }
  for (size_t i = 0; i < kps1.size(); i++) {
    kps1_norm.emplace_back(cv::Point2f((kps1[i].x - width / 2) / focal,
                                       (kps1[i].y - height / 2) / focal));
  }
  std::cout << "Finish inference mobilenetvlad" << std::endl;
  auto matches = sg_onnx.inference(kps0_norm, kps1_norm, local_desc0,
                                   local_desc1, scores0, scores1);
  std::cout << "Finish inference superglue" << std::endl;
  std::vector<cv::KeyPoint> _kps0;
  std::vector<cv::KeyPoint> _kps1;
  for (size_t i = 0; i < kps0.size(); i++) {
    cv::circle(img0, kps0[i], scores0[i] * 10, cv::Scalar(255, 0, 0), 1,
               cv::LINE_AA);
    _kps0.emplace_back(cv::KeyPoint(kps0[i].x, kps0[i].y, scores0[i]));
    kps0_norm.emplace_back(cv::Point2f(kps0[i].x / width, kps0[i].y / height));
  }
  for (size_t i = 0; i < kps1.size(); i++) {
    cv::circle(img1, kps1[i], scores1[i] * 10, cv::Scalar(255, 0, 0), 1,
               cv::LINE_AA);
    _kps1.emplace_back(cv::KeyPoint(kps1[i].x, kps1[i].y, scores1[i]));
    kps1_norm.emplace_back(cv::Point2f(kps0[i].x / width, kps0[i].y / height));
  }
  cv::hconcat(img0, img1, show);
  cv::imshow("Procesed", show);
  cv::waitKey(30);
  if (num_test > 0) {
    D2FrontEnd::TicToc tic;
    for (unsigned int i = 0; i < num_test; i++) {
      sp_onnx.inference(img_gray0, kps0, local_desc0, scores0);
    }
    double dt_sp = tic.toc();

    D2FrontEnd::TicToc tic2;
    for (unsigned int i = 0; i < num_test; i++) {
      netvlad_onnx.inference(img_gray0);
    }
    double dt_netvlad = tic2.toc();

    D2FrontEnd::TicToc tic3;
    for (unsigned int i = 0; i < num_test; i++) {
      sg_onnx.inference(kps0, kps1, local_desc0, local_desc1, scores0, scores1);
    }
    double dt_superglue = tic3.toc();

    printf("Run %d tests on SuperPoint, NetVLAD, SuperGlue\n", num_test);
    printf("SuperPoint: %.1fms\n", dt_sp / num_test);
    printf("NetVLAD: %.1fms\n", dt_netvlad / num_test);
    printf("SuperGlue: %.1fms\n", dt_superglue / num_test);
  }
  cv::drawMatches(img0, _kps0, img1, _kps1, matches, show);
  // for (auto match : matches) {
  //     //Draw match scores on image
  //     cv::putText(show, std::to_string(1-match.distance),
  //     kps0[match.queryIdx], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255,
  //     0), 1, cv::LINE_AA);
  // }
  cv::imshow("Matches", show);
  cv::waitKey(-1);
}