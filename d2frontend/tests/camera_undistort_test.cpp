#include "d2frontend/fisheye_undistort.h"
#include <camodocal/camera_models/CataCamera.h>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

namespace D2FrontEnd {
std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config);
}

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("image,i", po::value<std::string>()->default_value(""), "path of test image to undistort")
        ("calib,c", po::value<std::string>()->default_value(""), "path of camera config")
        ("name,n", po::value<std::string>()->default_value("cam0"), "name of camera")
        ("fov,f", po::value<double>()->default_value(190.0), "fov of camera");
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
    auto ret = D2FrontEnd::readCameraConfig(camera_name, config[camera_name]);
    printf("Read camera %s fov %f calib from file %s OK", camera_name.c_str(), fov, calib_path.c_str());

    //Initialize undistor
    FisheyeUndist undistort(ret.first, 0, fov, true, img.cols, img.cols/2);
    auto imgs = undistort.undist_all(img, true);
    cv::imshow("Undistort", imgs[0]);

    cv::waitKey(0);
}