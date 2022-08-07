#pragma once
#include <ceres/ceres.h>
#include "../utils.hpp"

namespace D2Common {
class ConsenusPoseFactor4D {
    Eigen::Matrix<double, 4, 4> _sqrt_inf;
    Eigen::Vector3d t_ref;
    double yaw_ref;
    bool norm_yaw;
public:
    ConsenusPoseFactor4D(Eigen::Vector3d _t_ref, double _yaw_ref, double rho_T, double rho_theta, bool _norm_yaw = false):
        norm_yaw(_norm_yaw), t_ref(_t_ref) {
        _sqrt_inf.setZero();
        if (norm_yaw) {
            yaw_ref = Utility::NormalizeAngle(_yaw_ref);
        } else {
            yaw_ref = _yaw_ref;
        }
        _sqrt_inf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * rho_T;
        _sqrt_inf(3, 3) = rho_theta;
    }

    template<typename T>
    bool operator()(const T* const p_a_ptr, T *_residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_a(p_a_ptr);
        Utility::poseError4D<T>(pos_a, p_a_ptr[3], 
            t_ref.template cast<T>(), (T)(yaw_ref), _sqrt_inf.template cast<T>(), _residual, norm_yaw);
        return true;
    }

    static ceres::CostFunction* Create(const Swarm::Pose & ref_pose, double rho_T, double rho_theta, bool norm_yaw = false) {
        return new ceres::AutoDiffCostFunction<ConsenusPoseFactor4D, 4, 4>(
            new ConsenusPoseFactor4D(ref_pose.pos(), ref_pose.yaw(), rho_T, rho_theta, norm_yaw));
    }

};
}