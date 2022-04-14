#include <Eigen/Eigen>
#include <swarm_msgs/Pose.h>

using namespace Eigen;

namespace D2VINS {
inline Quaterniond g2R(const Vector3d &g)
{
    Vector3d ng1 = g.normalized();
    Vector3d ng2{0, 0, 1.0};
    Quaterniond q0 = Quaterniond::FromTwoVectors(ng1, ng2);
    double yaw = quat2eulers(q0).z();
    q0 = eulers2quat(Vector3d{0, 0, -yaw}) * q0;
    return q0;
}
}