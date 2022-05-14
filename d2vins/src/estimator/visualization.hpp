#pragma once
#include <ros/ros.h>
#include <nav_msgs/Path.h>

namespace D2VINS {
class D2EstimatorState;
class D2Estimator;
class D2Visualization {
    D2Estimator * _estimator = nullptr;
    ros::Publisher odom_pub, imu_prop_pub, pcl_pub, margined_pcl, path_pub;
    nav_msgs::Path path;
public:
    void init(ros::NodeHandle & nh, D2Estimator * estimator);
    void postSolve();
};
}