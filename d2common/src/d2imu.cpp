#include <d2common/d2imu.h>
#include <d2common/d2vinsframe.h>
#include <d2common/integration_base.h>
#include <spdlog/spdlog.h>

namespace D2Common {

Vector3d IMUData::Gravity = Vector3d(0., 0., 9.805);
Eigen::Matrix<double, 18, 18> IntegrationBase::noise =
    Eigen::Matrix<double, 18, 18>::Zero();
size_t IMUBuffer::searchClosest(double t) const {
  const Guard lock(buf_lock);
  if (buf.size() == 0) {
    SPDLOG_WARN("IMUBuffer::searchClosest: empty buffer!");
    return 0;
  }
  if (buf.size() == 1) {
    return 0;
  }
  return searchClosest(t, 0, buf.size());
}

size_t IMUBuffer::searchClosest(double t, int i0, int i1) const {
  const double eps = 5e-4;
  const Guard lock(buf_lock);
  // printf("IMUBuffer::searchClosest: t=%f, i0=%d, i1=%d\n", t, i0, i1);
  if (i1 - i0 == 1) {
    return i0;
  }
  int i = (i0 + i1) / 2;
  if (i > buf.size()) {
    return i0;
  }
  if (buf[i].t > t - eps) {
    return searchClosest(t, i0, i);
  } else {
    return searchClosest(t, i, i1);
  }
}

IMUBuffer IMUBuffer::slice(int i0, int i1) const {
  const Guard lock(buf_lock);
  IMUBuffer ret;
  if (i0 > buf.size()) {
    return ret;
  }
  if (i1 + 1 > buf.size()) {
    ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.end());
  } else {
    ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.begin() + i1 + 1);
  }
  ret.t_last = buf.back().t;
  return ret;
}

void IMUBuffer::add(const IMUData& data) {
  const Guard lock(buf_lock);
  buf.emplace_back(data);
  t_last = data.t;
}

Vector3d IMUBuffer::mean_acc() const {
  const Guard lock(buf_lock);
  Vector3d acc_sum(0, 0, 0);
  for (auto& data : buf) {
    acc_sum += data.acc;
  }
  return acc_sum / size();
}

Vector3d IMUBuffer::mean_gyro() const {
  const Guard lock(buf_lock);
  Vector3d gyro_sum(0, 0, 0);
  for (auto& data : buf) {
    gyro_sum += data.gyro;
  }
  return gyro_sum / size();
}

size_t IMUBuffer::size() const {
  const Guard lock(buf_lock);
  return buf.size();
}

bool IMUBuffer::available(double t) const { return t_last > t; }

IMUBuffer IMUBuffer::pop(double t) {
  const Guard lock(buf_lock);
  if (buf.size() == 0) {
    return IMUBuffer();
  }
  auto i0 = searchClosest(t);
  IMUBuffer ret;
  if (i0 > 0) {
    ret.buf = std::vector<IMUData>(buf.begin(), buf.begin() + i0);
    ret.t_last = ret.buf.back().t;
    buf.erase(buf.begin(), buf.begin() + i0);
  }
  return ret;
}

IMUBuffer IMUBuffer::tail(double t) const {
  const Guard lock(buf_lock);
  if (buf.size() == 0) {
    return IMUBuffer();
  }
  auto i0 = searchClosest(t);
  IMUBuffer ret;
  ret.buf = std::vector<IMUData>(buf.begin() + i0, buf.end());
  ret.t_last = buf.back().t;
  return ret;
}

std::pair<IMUBuffer, int> IMUBuffer::periodIMU(double t0, double t1) const {
  const Guard lock(buf_lock);
  if (buf.size() == 0) {
    return std::make_pair(IMUBuffer(), 0);
  }
  auto i0 = searchClosest(t0);
  auto i1 = searchClosest(t1);
  return std::make_pair(slice(i0 + 1, i1 + 1), i1 + 1);
}

std::pair<IMUBuffer, int> IMUBuffer::periodIMU(int i0, double t1) const {
  const Guard lock(buf_lock);
  if (buf.size() == 0) {
    return std::make_pair(IMUBuffer(), 0);
  }
  auto i1 = searchClosest(t1, i0 + 1, buf.size());
  ;
  return std::make_pair(slice(i0 + 1, i1 + 1), i1 + 1);
}

Swarm::Odometry IMUBuffer::propagation(const VINSFramePtr& baseframe) const {
  return propagation(baseframe->odom, baseframe->Ba, baseframe->Bg);
}

Swarm::Odometry IMUBuffer::propagation(const Swarm::Odometry& prev_odom,
                                       const Vector3d& Ba,
                                       const Vector3d& Bg) const {
  const Guard lock(buf_lock);
  if (buf.size() == 0) {
    return prev_odom;
  }
  Swarm::Odometry odom = prev_odom;
  IMUData imu_last = buf[0];
  for (auto& imu : buf) {
    imu.propagation(odom, Ba, Bg, imu_last);
    imu_last = imu;
  }
  return odom;
}

void IMUData::propagation(Swarm::Odometry& odom, const Vector3d& Ba,
                          const Vector3d& Bg, const IMUData& imu_last) const {
  Vector3d un_acc_0 = odom.att() * (imu_last.acc - Ba) - Gravity;
  Vector3d un_gyr = 0.5 * (imu_last.gyro + this->gyro) - Bg;
  odom.att() = odom.att() * Utility::deltaQ(un_gyr * this->dt);
  odom.att().normalize();
  Vector3d un_acc_1 = odom.att() * (this->acc - Ba) - Gravity;
  Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  odom.pos() += this->dt * odom.vel() + 0.5 * this->dt * this->dt * un_acc;
  odom.vel() += this->dt * un_acc;
  odom.stamp = this->t;
}

}  // namespace D2Common