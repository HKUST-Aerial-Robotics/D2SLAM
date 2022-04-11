#include "swarm_loop/superpoint_tensorrt.h"
#include "swarm_loop/loop_defines.h"
#include "swarm_loop/mobilenetvlad_tensorrt.h"
#include "swarm_loop/utils.h"
#include "swarm_msgs/swarm_types.hpp"
using namespace Swarm;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        return -1;
    }

    std::cout << "Load Model from " << argv[1] << std::endl;
    std::cout << "Load Model from " << argv[2] << std::endl;
    
    std::string engine_path(argv[1]);
    std::string engine_path2(argv[2]);

    SuperPointTensorRT sp_trt(engine_path, "", "",  400, 208,0.012, true);
    MobileNetVLADTensorRT netvlad_trt(engine_path2, 400, 208, true);

    std::cout << "Load 2 Model success" << std::endl << " Loading image " << argv[3] << std::endl;

    cv::Mat img = cv::imread(argv[3]);
    cv::resize(img, img, cv::Size(400, 208));
    std::vector<float> local_desc;
    std::vector<cv::Point2f> kps;

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    TicToc tic;
    for (unsigned int i = 0; i < 1000; i ++) {
        sp_trt.inference(img_gray, kps, local_desc);
    }
    double dt = tic.toc();
    
    TicToc tic2;
    for (unsigned int i = 0; i < 1000; i ++) {
        netvlad_trt.inference(img_gray);
    }

    std::cout << "\nSuperpoint 1000 takes" << dt << std::endl;
    std::cout << "\nNetVLAD 1000 takes" << tic2.toc() << std::endl;
    
    for(auto pt : kps) {
        cv::circle(img, pt, 1, cv::Scalar(255, 0, 0), -1);
    }

    cv::resize(img, img, cv::Size(), 4, 4);
    cv::imshow("Image", img);
    cv::waitKey(-1);
}