#include <ros/ros.h>

#include "quadcam_depth_est.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "quadcam_depth_est");
  ros::NodeHandle nh("~");
  D2QuadCamDepthEst::QuadCamDepthEst quadcam_depth_est(nh);
  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();
  return 0;
}