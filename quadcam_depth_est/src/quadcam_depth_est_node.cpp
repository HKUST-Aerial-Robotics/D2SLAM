#include <ros/ros.h>
#include "quadcam_depth_est_trt.hpp"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "quadcam_depth_est");
    ros::NodeHandle nh("~");
    // D2QuadCamDepthEst::QuadCamDepthEst quadcam_depth_est(nh);
    // ros::MultiThreadedSpinner spinner(3);
    D2QuadCamDepthEst::QuadcamDepthEstTrt quadcam_depth_est(nh);
    quadcam_depth_est.startAllService();
    ros::spin();
    quadcam_depth_est.stopAllService();
    usleep(100);
    return 0;
}