#pragma once
#include "sensor_msgs/Imu.h"
#include "swarm_msgs/Pose.h"
#include <swarm_msgs/Odometry.h>
#include <mutex>
#include <swarm_msgs/lcm_gen/IMUData_t.hpp>
#include <swarm_msgs/swarm_lcm_converter.hpp>

namespace D2Common {
typedef std::lock_guard<std::recursive_mutex> Guard;

struct VINSFrame;
using VINSFramePtr = std::shared_ptr<VINSFrame>;

struct IMUData {
    static Vector3d Gravity;

    double t = 0.0;
    double dt = 0.0;
    Vector3d acc;
    Vector3d gyro;

    IMUData(): acc(0.0, 0.0, 0.0),gyro(0.0, 0.0, 0.0) {}
    
    IMUData(const sensor_msgs::Imu & imu):
        t(imu.header.stamp.toSec()),
        gyro(imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z),
        acc(imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z)
    {}

    IMUData(const IMUData_t & imu):
        dt(imu.dt),
        acc(imu.acc.x, imu.acc.y, imu.acc.z),
        gyro(imu.gyro.x, imu.gyro.y, imu.gyro.z) {
        t = toROSTime(imu.timestamp).toSec();
    }

    IMUData_t toLCM() const { 
        IMUData_t imu;
        imu.timestamp = toLCMTime(ros::Time(t));
        imu.dt = dt;
        imu.gyro.x = gyro.x();
        imu.gyro.y = gyro.y();
        imu.gyro.z = gyro.z();
        imu.acc.x = acc.x();
        imu.acc.y = acc.y();
        imu.acc.z = acc.z();
        return imu;
    }
    
    void propagation(Swarm::Odometry & odom, const Vector3d & Ba, const Vector3d & Bg, const IMUData & imu_last) const;
};


class IMUBuffer {
protected:
    size_t searchClosest(double t) const;
    //Search [i0, i1)
    size_t searchClosest(double t, int i0, int i1) const;
    IMUBuffer slice(int i0, int i1) const;
    mutable std::recursive_mutex buf_lock;
public:
    std::vector<IMUData> buf;
    double t_last = 0.0;
    
    IMUBuffer(const IMUBuffer & _buf)
        : buf(_buf.buf), t_last(_buf.t_last) {}
    IMUBuffer(const std::vector<IMUData> & _buf) {
        for (auto data : _buf) {
            add(data);
        }
    }
    IMUBuffer() {}
    
    void operator=(const IMUBuffer & _buf) {
        buf = _buf.buf;
        t_last = _buf.t_last;
    }


    void add(const IMUData & data);

    Vector3d mean_acc() const;

    Vector3d mean_gyro() const;

    size_t size() const;

    bool available(double t) const;

    IMUBuffer pop(double t);

    IMUBuffer tail(double t) const;

    //Return imu buf and last data's index
    std::pair<IMUBuffer, int> periodIMU(double t0, double t1) const;
    std::pair<IMUBuffer, int> periodIMU(int i0, double t1) const;

    Swarm::Odometry propagation(const Swarm::Odometry & odom, const Vector3d & Ba, const Vector3d & Bg) const;
    Swarm::Odometry propagation(const VINSFramePtr & baseframe) const;
    IMUData operator[](int i) const {
        return buf.at(i);
    }
    
    IMUData & operator[](int i) {
        return buf.at(i);
    }

    void clear() {
        buf.clear();
        t_last = 0.0;
    }
};
}