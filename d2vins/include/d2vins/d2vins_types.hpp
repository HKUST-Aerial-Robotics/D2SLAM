#pragma once

#include "d2frontend/d2frontend_types.h"
#include "sensor_msgs/Imu.h"

namespace D2VINS {
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

    struct IMUBuffer
    {
        std::vector<IMUData> buf;
        double t_last = 0.0;
        void add(const IMUData & data) {
            buf.emplace_back(data);
            t_last = data.t;
        }

        Vector3d mean_acc() const {
            Vector3d acc_sum(0, 0, 0);
            for (auto & data : buf) {
                acc_sum += data.acc;
            }
            return acc_sum/size();
        }

        Vector3d mean_gyro() const {
            Vector3d gyro_sum(0, 0, 0);
            for (auto & data : buf) {
                gyro_sum += data.gyro;
            }
            return gyro_sum/size();
        }


        size_t size() const {
            return buf.size();
        }

        bool avaiable(double t) const {
            return t_last > t;
        }
    };

    
}