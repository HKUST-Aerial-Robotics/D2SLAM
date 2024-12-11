#include <camodocal/camera_models/CataCamera.h>
#include <yaml-cpp/yaml.h>

#include <boost/program_options.hpp>

#include "d2common/fisheye_undistort.h"
#include "d2frontend/d2frontend_params.h"

// namespace D2FrontEnd {
// std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(
//     const std::string& camera_name, const YAML::Node& config, int32_t extrinsic_parameter_type = 1 );
// }

using namespace D2FrontEnd;
using D2Common::FisheyeUndist;

int main(int argc, char** argv) {
  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "image,i", po::value<std::string>()->default_value(""),
      "path of test image to undistort")(
      "calib,c", po::value<std::string>()->default_value(""),
      "path of camera config")("name,n",
                               po::value<std::string>()->default_value("cam0"),
                               "name of camera")(
      "fov,f", po::value<double>()->default_value(190.0), "fov of camera");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  auto image_path = vm["image"].as<std::string>();
  auto calib_path = vm["calib"].as<std::string>();
  auto camera_name = vm["name"].as<std::string>();
  auto fov = vm["fov"].as<double>();

  cv::Mat img = cv::imread(image_path, cv::IMREAD_ANYCOLOR);
  cv::imshow("Raw", img);
  cv::waitKey(1);
  printf("Read image file %s OK", image_path.c_str());
  YAML::Node config = YAML::LoadFile(calib_path);
  auto ret = D2FrontendParams::readCameraConfig(camera_name, config[camera_name]);
  printf("Read camera %s fov %f calib from file %s OK", camera_name.c_str(),
         fov, calib_path.c_str());

  // Initialize undistort
  int undistort_width = img.cols;
  int undistort_height = img.cols / 2;
  int pinhole2_height = img.cols * 0.75;
  FisheyeUndist undistort(ret.first, 0, fov, FisheyeUndist::UndistortCylindrical, undistort_width,
                          undistort_height);
  auto imgs = undistort.undist_all(img, true);
  cv::imshow("UndistortCylindrical", imgs[0]);

#ifdef USE_CUDA
  auto img_cuda = undistort.undist_id_cuda(img, 0);
  cv::Mat img_cpu(img_cuda);
  cv::imshow("UndistortCylindrical_cuda", img_cpu);
#endif

  FisheyeUndist undistort5(ret.first, 0, fov, FisheyeUndist::UndistortPinhole5, undistort_width,
                           undistort_height);
  imgs = undistort5.undist_all(img, true);
  cv::imshow("UndistortPinhole5", imgs[0]);

  FisheyeUndist undistort2(ret.first, 0, fov, FisheyeUndist::UndistortPinhole2, undistort_width,
                           pinhole2_height);
  imgs = undistort2.undist_all(img, true);
  cv::hconcat(imgs[0], imgs[1], img);
  cv::imshow("UndistortPinhole2", img);

  double err = 0;
  for (int i = 0; i < undistort_width; i++) {
    for (int j = 0; j < undistort_height; j++) {
      Vector2d p(i, j), p_rep;
      Vector3d p3d;
      undistort.cam_top->liftProjective(p, p3d);
      undistort.cam_top->spaceToPlane(p3d, p_rep);
      err = err + (p_rep - p).norm();
    }
  }
  printf("Undistort error %f\n", err);

  cv::waitKey(0);
}