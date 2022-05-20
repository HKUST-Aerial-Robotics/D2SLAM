#pragma once
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <Eigen/Eigen>

namespace D2VINS {
class D2EstimatorState;
class D2Estimator;
class D2Visualization {
    D2Estimator * _estimator = nullptr;
    ros::Publisher odom_pub, imu_prop_pub, pcl_pub, margined_pcl, path_pub;
    ros::Publisher sld_win_pub;
    nav_msgs::Path path;
    std::vector<Eigen::Vector3d> drone_colors;
    double display_alpha = 0.5;
public:
    D2Visualization();
    void init(ros::NodeHandle & nh, D2Estimator * estimator);
    void postSolve();
};
}