#include "d2common/fisheye_undistort.h"
#include <camodocal/camera_models/CataCamera.h>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>
#include "../src/quadcam_depth_est.hpp"
#include "../src/hitnet_onnx.hpp"

namespace D2FrontEnd {
std::pair<camodocal::CameraPtr, Swarm::Pose> readCameraConfig(const std::string & camera_name, const YAML::Node & config);
}

using namespace D2FrontEnd;
using namespace D2QuadCamDepthEst;
using D2Common::FisheyeUndist;

void drawGrid(cv::Mat & show, int num_rows = 50) {
    for (int i = 0; i < num_rows; i ++ ) {
        cv::Scalar c(0, 0, 0);
        if (i % 5 == 0) {
            c = cv::Scalar(0, 255, 0);
        }
        cv::line(show, cv::Point(i*show.cols/num_rows, 0), cv::Point(i*show.cols/num_rows, show.rows - 1), c, 1);
    }
}

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
        ("fov,f", po::value<double>()->default_value(190.0), "fov of camera")
        ("engine,e", po::value<std::string>()->default_value(""), "engine of onnx");
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
    auto engine_path = vm["engine"].as<std::string>();

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
    FisheyeUndist undistort2_0(ret_left.first, 0, fov, FisheyeUndist::UndistortPinhole2, width, height);
    FisheyeUndist undistort2_1(ret_right.first, 0, fov, FisheyeUndist::UndistortPinhole2, width, height);
    auto imgs_left = undistort2_0.undist_all(img_l, true);
    auto imgs_right = undistort2_1.undist_all(img_r, true);
    cv::vconcat(imgs_left[idx0], imgs_right[idx1], show);
    drawGrid(show);
    cv::namedWindow("RawStereoImgs", cv::WINDOW_NORMAL|cv::WINDOW_GUI_EXPANDED);
    cv::imshow("RawStereoImgs", show);
    VirtualStereo virtual_stereo(0, 1, Swarm::Pose(), 
        &undistort2_0, &undistort2_1, idx0, idx1, nullptr, nullptr);
    auto rect_imgs = virtual_stereo.rectifyImage(img_l, img_r);
    cv::Mat rect_l(rect_imgs[0]), rect_r(rect_imgs[1]);
    show.release();
    cv::hconcat(rect_l, rect_r, show);
    int num_rows = 10;
    for (int i = 0; i < num_rows; i ++ ) {
        cv::line(show, cv::Point(0, i*show.rows/num_rows), cv::Point(show.cols - 1,i*show.rows/num_rows), cv::Scalar(0, 255, 0), 1);
    }
    cv::namedWindow("Rectified Images", cv::WINDOW_NORMAL|cv::WINDOW_GUI_EXPANDED);
    cv::imshow("Rectified Images", show);
    cv::vconcat(rect_l, rect_r, show);
    //Draw vertical lines
    drawGrid(show);
    cv::namedWindow("Rectified Images2", cv::WINDOW_NORMAL|cv::WINDOW_GUI_EXPANDED);
    cv::imshow("Rectified Images2", show);
    cv::imwrite("rect_l.png", rect_l);
    cv::imwrite("rect_r.png", rect_r);

    auto disp_cuda = virtual_stereo.estimateDisparityOCV(rect_l, rect_r);
    cv::Mat disp_show(disp_cuda);
    double min_val=0, max_val=0;
    cv::normalize(disp_show, disp_show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(disp_show, disp_show, cv::COLORMAP_JET);
    cv::imshow("DisparityOCV", disp_show);

    if (engine_path != "") {
        HitnetONNX hitnet_onnx(engine_path, 320, 240);
        auto disp = hitnet_onnx.inference(rect_l, rect_r);
        TicToc t;
        for (int i = 0 ; i < 100; i ++) {
            disp = hitnet_onnx.inference(rect_l, rect_r);
        }
        printf("Inference time: %.1fms\n", t.toc()/100);
        double min_val=0, max_val=0;
        cv::normalize(disp, disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(disp, disp, cv::COLORMAP_JET);
        cv::imshow("DisparityONNX", disp);
    }
    cv::waitKey(0);
}