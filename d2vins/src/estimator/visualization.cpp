#include "visualization.hpp"
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud.h>
#include "d2estimator.hpp"

namespace D2VINS {

sensor_msgs::PointCloud toPointCloud(const std::vector<D2FrontEnd::LandmarkPerId> landmarks);

void D2Visualization::init(ros::NodeHandle & nh, D2Estimator * estimator) {
    pcl_pub = nh.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    odom_pub = nh.advertise<nav_msgs::Odometry>("odometry", 1000);
    imu_prop_pub = nh.advertise<nav_msgs::Odometry>("imu_propagation", 1000);
    _estimator = estimator;
}

void D2Visualization::postSolve() {
    auto odom = _estimator->getOdometry().toRos();
    odom_pub.publish(odom);
    auto imu_prop = _estimator->getImuPropagation().toRos();
    imu_prop_pub.publish(imu_prop);
    auto pcl = _estimator->getState().getInitializedLandmarks();
    pcl_pub.publish(toPointCloud(pcl));
}

sensor_msgs::PointCloud toPointCloud(const std::vector<D2FrontEnd::LandmarkPerId> landmarks) {
    sensor_msgs::PointCloud pcl;
    pcl.header.frame_id = "world";
    pcl.points.resize(landmarks.size());
    pcl.channels.resize(3);
    pcl.channels[0].name = "rgb";
    pcl.channels[0].values.resize(landmarks.size());
    for (int i = 0; i < landmarks.size(); i++) {
        pcl.points[i].x = landmarks[i].position.x();
        pcl.points[i].y = landmarks[i].position.y();
        pcl.points[i].z = landmarks[i].position.z();
        Vector3i color(255., 255., 0.);
        if (landmarks[i].flag == D2FrontEnd::LandmarkFlag::ESTIMATED) {
            //set color to green
            color = Vector3i(0., 255., 0.);
        } 
        uint32_t hex_r = (0xff & color(0)) << 16;
        uint32_t hex_g = (0xff & color(1)) << 8;
        uint32_t hex_b = (0xff & color(2));
        uint32_t hex_rgb = hex_r | hex_g | hex_b;
        memcpy(pcl.channels[0].values.data() + i, &hex_rgb, sizeof(float));
    }
    return pcl;
}
}