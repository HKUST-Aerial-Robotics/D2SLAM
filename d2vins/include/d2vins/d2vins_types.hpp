#pragma once

#include "d2frontend/d2frontend_types.h"
#include "sensor_msgs/Imu.h"
#include "d2vins_params.hpp"
#include "utils.hpp"
#include "factors/imu_factor.h"

namespace D2VINS {
struct VINSFrame {
    double stamp;
    int frame_id;
    Swarm::Pose pose;
    Vector3d V; //Velocity
    Vector3d Ba; // bias of acc
    Vector3d Bg; //bias of gyro
    VINSFrame():V(0., 0., 0.), Ba(0., 0., 0.), Bg(0., 0., 0.)
    {}
    
    VINSFrame(const D2Frontend::VisualImageDescArray & frame):
        stamp(frame.stamp),
        frame_id(frame.frame_id),
        V(0., 0., 0.), Ba(0., 0., 0.), Bg(0., 0., 0.) {
    }
};

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
    size_t searchClosest(double t) const {
        if (buf.size() == 1) {
            return 0;
        }
        
        for (size_t i = 0; i < buf.size() - 1; i ++ ) {
            if (fabs(buf[i].t  - t) < fabs(buf[i + 1].t  - t)) {
                return i;
            }
        }

        return buf.size() - 1;
    }

    IMUBuffer slice(int i0, int i1) const {
        IMUBuffer ret;
        ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.begin() + i1);
        ret.t_last = buf.back().t;
        return ret;
    }
public:
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

    IMUBuffer pop(double t) {
        auto i0 = searchClosest(t);
        IMUBuffer ret;
        ret.buf = std::vector<IMUData>(buf.begin(), buf.begin() + i0);
        ret.t_last = ret.buf.back().t;
        buf.erase(buf.begin(), buf.begin() + i0);
        return ret;
    }

    IMUBuffer back(double t) const {
        auto i0 = searchClosest(t);
        IMUBuffer ret;
        ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.end());
        ret.t_last = buf.back().t;
        return ret;
    }

    IMUBuffer periodIMU(double t0, double t1) const {
        auto i0 = searchClosest(t0);
        auto i1 = searchClosest(t1);
        return slice(i0, i1);
    }

    std::pair<Swarm::Pose, Vector3d> propagation(const Swarm::Pose & p0, 
            const Vector3d & V0, const Vector3d & Ba, const Vector3d & Bg) const {
        if(buf.size() == 0) {
            return std::make_pair(p0, V0);
        }
        Vector3d acc_last = buf[0].acc;
        Vector3d gyro_last = buf[0].gyro;

        Swarm::Pose pret = p0;
        Vector3d Vs = V0;
        for (auto & imu: buf) {
            Vector3d un_acc_0 = pret.att() * (acc_last - Ba) - Gravity;
            Vector3d un_gyr = 0.5 * (gyro_last + imu.gyro) - Bg;
            pret.att() = pret.att() * Utility::deltaQ(un_gyr * imu.dt);
            Vector3d un_acc_1 = pret.att() * (imu.acc - Ba) - Gravity;
            Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            pret.pos() += imu.dt * V0 + 0.5 * imu.dt * imu.dt * un_acc;
            Vs += imu.dt * un_acc;
            acc_last = imu.acc;
            gyro_last = imu.gyro;
        }

        return std::make_pair(pret, Vs);
    }

    std::pair<Swarm::Pose, Vector3d> propagation(const VINSFrame & baseframe) const {
        return propagation(baseframe.pose, baseframe.V, baseframe.Ba, baseframe.Bg);
    }
};

    
}