#include "d2common/fisheye_undistort.h"
#include <camodocal/camera_models/CataCamera.h>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>
#include "../src/quadcam_depth_est.hpp"

namespace D2FrontEnd {
std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config);
}

using namespace D2FrontEnd;
using namespace D2QuadCamDepthEst;
using D2Common::FisheyeUndist;

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("image_l,l", po::value<std::string>()->default_value(""), "path of test image l")
        ("image_r,r", po::value<std::string>()->default_value(""), "path of test image r")
        ("calib,c", po::value<std::string>()->default_value(""), "path of camera config")
        ("name0", po::value<std::string>()->default_value("cam3"), "name of camera_left")
        ("name1", po::value<std::string>()->default_value("cam2"), "name of camera_right")
        ("idx0", po::value<int>()->default_value(1), "idx of camera_left")
        ("idx1", po::value<int>()->default_value(0), "idx of camera_right")
        ("width,w", po::value<int>()->default_value(600), "width of camera")
        ("height,h", po::value<int>()->default_value(300), "height of camera")
        ("fov,f", po::value<double>()->default_value(190.0), "fov of camera");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    auto image_path_l = vm["image_l"].as<std::string>();
    auto image_path_r = vm["image_r"].as<std::string>();
    auto calib_path = vm["calib"].as<std::string>();
    auto camera_name0 = vm["name0"].as<std::string>();
    auto camera_name1 = vm["name1"].as<std::string>();
    auto fov = vm["fov"].as<double>();
    int idx0 = vm["idx0"].as<int>();
    int idx1 = vm["idx1"].as<int>();
    int width = vm["width"].as<int>();
    int height = vm["height"].as<int>();

    cv::Mat img_l = cv::imread(image_path_l, cv::IMREAD_ANYCOLOR);
    cv::Mat img_r = cv::imread(image_path_r, cv::IMREAD_ANYCOLOR);
    cv::Mat show;
    cv::hconcat(img_l, img_r, show);
    cv::imshow("Raw", show);
    cv::waitKey(1);
    printf("Read image file %s %s OK", image_path_l.c_str(), image_path_r.c_str());
    YAML::Node config = YAML::LoadFile(calib_path);
    auto ret_left = D2FrontEnd::readCameraConfig(camera_name0, config[camera_name0]);
    auto ret_right = D2FrontEnd::readCameraConfig(camera_name1, config[camera_name1]);

    //Initialize undistort
    FisheyeUndist undistort2_0(ret_left.first, 0, fov, true, FisheyeUndist::UndistortPinhole2, width, height);
    FisheyeUndist undistort2_1(ret_right.first, 0, fov, true, FisheyeUndist::UndistortPinhole2, width, height);
    auto imgs_left = undistort2_0.undist_all(img_l, true);
    auto imgs_right = undistort2_1.undist_all(img_r, true);
    cv::hconcat(imgs_left[idx0], imgs_right[idx1], show);
    cv::imshow("RawStereoImgs", show);

    VirtualStereo virtual_stereo(0, 1, ret_left.second, ret_right.second, 
        &undistort2_0, &undistort2_1, idx0, idx1);
    auto rect_imgs = virtual_stereo.rectifyImage(img_l, img_r);
    cv::Mat rect_l(rect_imgs[0]), rect_r(rect_imgs[1]);
    show.release();
    cv::hconcat(rect_l, rect_r, show);
    int num_rows = 10;
    for (int i = 0; i < show.rows/num_rows; i ++ ) {
        cv::line(show, cv::Point(0, i*show.rows/num_rows), cv::Point(show.cols - 1,i*show.rows/num_rows), cv::Scalar(0, 255, 0), 1);
    }
    cv::imshow("Rectified Images", show);
    cv::vconcat(rect_l, rect_r, show);
    //Draw vertical lines
    for (int i = 0; i < show.cols/num_rows; i ++ ) {
        cv::line(show, cv::Point(i*show.cols/num_rows, 0), cv::Point(i*show.cols/num_rows, show.rows - 1), cv::Scalar(0, 255, 0), 1);
    }
    cv::imshow("Rectified Images2", show);
    cv::imwrite("rect_l.png", rect_l);
    cv::imwrite("rect_r.png", rect_r);

    auto disp_cuda = virtual_stereo.estimateDisparityOCV(rect_l, rect_r);
    cv::Mat disp_show(disp_cuda);
    double min_val=0, max_val=0;
    cv::minMaxLoc(disp_show, &min_val, &max_val, NULL, NULL);
    printf("Disp max %f min %f\n", max_val, min_val);
    disp_show.convertTo(disp_show, CV_8U, 255.0/max_val);
    cv::applyColorMap(disp_show, disp_show, cv::COLORMAP_RAINBOW);
    // printf("AFT Disp max %f min %f\n", max_val, min_val);
    cv::imshow("Disparity", disp_show);


    cv::waitKey(0);
}