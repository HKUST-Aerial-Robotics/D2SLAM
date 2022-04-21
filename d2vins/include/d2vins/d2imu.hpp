#pragma once
#include "sensor_msgs/Imu.h"
#include "swarm_msgs/Pose.h"
#include <swarm_msgs/Odometry.h>

namespace D2VINS {

struct VINSFrame;

struct IMUData {
    double t = 0.0;
    double dt = 0.0;
    Vector3d acc;
    Vector3d gyro;
    IMUData(): acc(0.0, 0.0, 0.0),gyro(0.0, 0.0, 0.0){}
    IMUData(const sensor_msgs::Imu & imu):
        t(imu.header.stamp.toSec()),
        gyro(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z),
        acc(imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z)
    {}
};

class IMUBuffer
{
protected:
    size_t searchClosest(double t) const;
    //Search [i0, i1)
    size_t searchClosest(double t, int i0, int i1) const;
    IMUBuffer slice(int i0, int i1) const;
public:
    std::vector<IMUData> buf;

    double t_last = 0.0;
    void add(const IMUData & data);

    Vector3d mean_acc() const;

    Vector3d mean_gyro() const;

    size_t size() const;

    bool available(double t) const;

    IMUBuffer pop(double t);

    IMUBuffer back(double t) const;

    IMUBuffer periodIMU(double t0, double t1) const;

    Swarm::Odometry propagation(const Swarm::Odometry & odom, const Vector3d & Ba, const Vector3d & Bg) const;
    Swarm::Odometry propagation(const VINSFrame & baseframe) const;
    IMUData operator[](int i) const {
        return buf.at(i);
    }
    
    IMUData & operator[](int i) {
        return buf.at(i);
    }
};
}