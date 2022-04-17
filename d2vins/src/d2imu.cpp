#include <d2vins/d2imu.hpp>
#include <d2vins/d2vins_types.hpp>

namespace D2VINS {

std::pair<Swarm::Pose, Vector3d> IMUBuffer::propagation(const VINSFrame & baseframe) const {
    return propagation(baseframe.pose, baseframe.V, baseframe.Ba, baseframe.Bg);
}

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

bool IMUBuffer::avaiable(double t) const {
    return t_last > t;
}

IMUBuffer IMUBuffer::pop(double t) {
    auto i0 = searchClosest(t);
    IMUBuffer ret;
    ret.buf = std::vector<IMUData>(buf.begin(), buf.begin() + i0);
    ret.t_last = ret.buf.back().t;
    buf.erase(buf.begin(), buf.begin() + i0);
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

std::pair<Swarm::Pose, Vector3d> IMUBuffer::propagation(const Swarm::Pose & p0, 
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

}