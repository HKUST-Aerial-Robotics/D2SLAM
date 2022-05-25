#include <d2common/d2imu.h>
#include <d2common/d2vinsframe.h>
#include <d2common/integration_base.h>
namespace D2Common {

Vector3d IMUBuffer::Gravity = Vector3d(0., 0., 9.805);
Eigen::Matrix<double, 18, 18> IntegrationBase::noise = Eigen::Matrix<double, 18, 18>::Zero();

size_t IMUBuffer::searchClosest(double t) const {
    const Guard lock(buf_lock);
    if (buf.size() == 0) {
        printf("IMUBuffer::searchClosest: empty buffer\n");
        return 0;
    }
    if (buf.size() == 1) {
        return 0;
    }
    return searchClosest(t, 0, buf.size());
}

size_t IMUBuffer::searchClosest(double t, int i0, int i1) const {
    const Guard lock(buf_lock);
    if (i1 - i0 == 1) {
        return i0;
    }
    if (i1 - i0 == 2) {
        //select i0 or i0 + 1
        if (std::abs(buf[i0 + 1].t - t) < std::abs(buf[i0].t - t)) {
            return i0 + 1;
        } else {
            return i0;
        }
    }

    int i = (i0 + i1) / 2;
    if (buf[i].t > t) {
        return searchClosest(t, i0, std::min(i + 1, i1));
    } else {
        return searchClosest(t, i, i1);
    }
}

IMUBuffer IMUBuffer::slice(int i0, int i1) const {
    const Guard lock(buf_lock);
    IMUBuffer ret;
    ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.begin() + i1 + 1);
    ret.t_last = buf.back().t;
    return ret;
}

void IMUBuffer::add(const IMUData & data) {
    const Guard lock(buf_lock);
    buf.emplace_back(data);
    t_last = data.t;
    buf_lock.unlock();
}

Vector3d IMUBuffer::mean_acc() const {
    const Guard lock(buf_lock);
    Vector3d acc_sum(0, 0, 0);
    for (auto & data : buf) {
        acc_sum += data.acc;
    }
    return acc_sum/size();
}

Vector3d IMUBuffer::mean_gyro() const {
    const Guard lock(buf_lock);
    Vector3d gyro_sum(0, 0, 0);
    for (auto & data : buf) {
        gyro_sum += data.gyro;
    }
    return gyro_sum/size();
}

size_t IMUBuffer::size() const {
    const Guard lock(buf_lock);
    return buf.size();
}

bool IMUBuffer::available(double t) const {
    return t_last > t;
}

IMUBuffer IMUBuffer::pop(double t) {
    const Guard lock(buf_lock);
    if (buf.size() == 0){
        return IMUBuffer();
    }
    auto i0 = searchClosest(t);
    IMUBuffer ret;
    if (i0 > 0) {
        ret.buf = std::vector<IMUData>(buf.begin(), buf.begin() + i0 + 1);
        ret.t_last = ret.buf.back().t;
        buf.erase(buf.begin(), buf.begin() + i0 + 1);
    }
    return ret;
}

IMUBuffer IMUBuffer::back(double t) const {
    const Guard lock(buf_lock);
    if (buf.size() == 0){
        return IMUBuffer();
    }
    auto i0 = searchClosest(t);
    IMUBuffer ret;
    ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.end());
    ret.t_last = buf.back().t;
    return ret;
}

IMUBuffer IMUBuffer::periodIMU(double t0, double t1) const {
    const Guard lock(buf_lock);
    if (buf.size() == 0){
        return IMUBuffer();
    }
    auto i0 = searchClosest(t0);
    auto i1 = searchClosest(t1);
    return slice(i0, i1);
}

Swarm::Odometry IMUBuffer::propagation(const VINSFrame & baseframe) const {
    return propagation(baseframe.odom, baseframe.Ba, baseframe.Bg);
}

Swarm::Odometry IMUBuffer::propagation(const Swarm::Odometry & prev_odom, const Vector3d & Ba, const Vector3d & Bg) const {
    const Guard lock(buf_lock);
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