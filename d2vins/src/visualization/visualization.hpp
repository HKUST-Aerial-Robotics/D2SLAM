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
    std::map<int, ros::Publisher> path_pubs, odom_pubs;
    ros::Publisher sld_win_pub;
    std::map<int, nav_msgs::Path> paths;
    double display_alpha = 0.5;
    ros::NodeHandle * _nh = nullptr;
public:
    D2Visualization();
    void init(ros::NodeHandle & nh, D2Estimator * estimator);
    void postSolve();
    static std::vector<Eigen::Vector3d> drone_colors;
};
}