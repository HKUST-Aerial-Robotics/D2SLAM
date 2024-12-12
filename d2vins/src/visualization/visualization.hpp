#pragma once
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <Eigen/Eigen>
#include <d2common/d2vinsframe.h>

namespace D2VINS {
class D2EstimatorState;
class D2Estimator;
class D2Visualization {
    D2Estimator * _estimator = nullptr;
    ros::Publisher odom_pub, imu_prop_pub, pcl_pub, margined_pcl, path_pub;
    ros::Publisher frame_pub_local, frame_pub_remote;
    std::vector<ros::Publisher> camera_pose_pubs;
    std::map<int, ros::Publisher> path_pubs, odom_pubs;
    ros::Publisher sld_win_pub;
    ros::Publisher cam_pub;
    std::map<int, nav_msgs::Path> paths;
    double display_alpha = 0.5;
    ros::NodeHandle * _nh = nullptr;
    std::map<int, std::ofstream> csv_output_files;
    tf::TransformBroadcaster br;
public:
    D2Visualization();
    void init(ros::NodeHandle & nh, D2Estimator * estimator);
    void postSolve();
    void pubFrame(const std::shared_ptr<D2Common::VINSFrame>& frame);
    void pubIMUProp(const Swarm::Odometry & odom);
    void pubOdometry(int drone_id, const Swarm::Odometry & odom);
    static std::vector<Eigen::Vector3d> drone_colors;
};
}