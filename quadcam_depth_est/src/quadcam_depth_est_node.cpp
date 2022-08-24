#include <ros/ros.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "quadcam_depth_est_node");
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    return 0;
}