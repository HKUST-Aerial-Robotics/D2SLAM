#include <d2vins/d2imu.hpp>
#include <d2vins/d2vins_types.hpp>

namespace D2VINS {
size_t IMUBuffer::searchClosest(double t) const {
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

IMUBuffer IMUBuffer::slice(int i0, int i1) const {
    IMUBuffer ret;
    ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.begin() + i1);
    ret.t_last = buf.back().t;
    return ret;
}

void IMUBuffer::add(const IMUData & data) {
    buf.emplace_back(data);
    t_last = data.t;
}

Vector3d IMUBuffer::mean_acc() const {
    Vector3d acc_sum(0, 0, 0);
    for (auto & data : buf) {
        acc_sum += data.acc;
    }
    return acc_sum/size();
}

Vector3d IMUBuffer::mean_gyro() const {
    Vector3d gyro_sum(0, 0, 0);
    for (auto & data : buf) {
        gyro_sum += data.gyro;
    }
    return gyro_sum/size();
}

size_t IMUBuffer::size() const {
    return buf.size();
}

bool IMUBuffer::available(double t) const {
    return t_last > t;
}

IMUBuffer IMUBuffer::pop(double t) {
    auto i0 = searchClosest(t);
    IMUBuffer ret;
    if (i0 > 0) {
        ret.buf = std::vector<IMUData>(buf.begin(), buf.begin() + i0);
        ret.t_last = ret.buf.back().t;
        buf.erase(buf.begin(), buf.begin() + i0);
    }
    return ret;
}

IMUBuffer IMUBuffer::back(double t) const {
    auto i0 = searchClosest(t);
    IMUBuffer ret;
    ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.end());
    ret.t_last = buf.back().t;
    return ret;
}

IMUBuffer IMUBuffer::periodIMU(double t0, double t1) const {
    auto i0 = searchClosest(t0);
    auto i1 = searchClosest(t1);
    return slice(i0, i1);
}

Swarm::Odometry IMUBuffer::propagation(const VINSFrame & baseframe) const {
    return propagation(baseframe.odom, baseframe.Ba, baseframe.Bg);
}

Swarm::Odometry IMUBuffer::propagation(const Swarm::Odometry & prev_odom, const Vector3d & Ba, const Vector3d & Bg) const {
    if(buf.size() == 0) {
        return prev_odom;
    }
    Vector3d acc_last = buf[0].acc;
    Vector3d gyro_last = buf[0].gyro;

    Swarm::Odometry odom = prev_odom;
    for (auto & imu: buf) {
        Vector3d un_acc_0 = prev_odom.att() * (acc_last - Ba) - Gravity;
        Vector3d un_gyr = 0.5 * (gyro_last + imu.gyro) - Bg;
        odom.att() = odom.att() * Utility::deltaQ(un_gyr * imu.dt);
        odom.att().normalize();
        Vector3d un_acc_1 = odom.att() * (imu.acc - Ba) - Gravity;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        odom.pos() += imu.dt * odom.vel() + 0.5 * imu.dt * imu.dt * un_acc;
        odom.vel() += imu.dt * un_acc;
        acc_last = imu.acc;
        gyro_last = imu.gyro;
    }
    return odom;
}

}