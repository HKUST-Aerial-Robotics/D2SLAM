#pragma once
#include <ceres/ceres.h>
#include "../utils.hpp"

namespace D2Common {
class ConsenusPoseFactor4D {
    Eigen::Matrix<double, 4, 4> _sqrt_inf;
    Eigen::Vector3d t_ref;
    double yaw_ref;
public:
    ConsenusPoseFactor4D(Eigen::Vector3d _t_ref, double _yaw_ref, double rho_T, double rho_theta) {
        _sqrt_inf.setZero();
        _sqrt_inf.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * rho_T;
        _sqrt_inf(3, 3) = rho_theta;
    }

    template<typename T>
    bool operator()(const T* const p_a_ptr, T *_residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_a(p_a_ptr);
        Utility::poseError4D<T>(pos_a, p_a_ptr[3], 
            t_ref.template cast<T>(), (T)(yaw_ref), _sqrt_inf.template cast<T>(), _residual);
        return true;
    }

    static ceres::CostFunction* Create(const Swarm::Pose & ref_pose, double rho_T, double rho_theta) {
        return new ceres::AutoDiffCostFunction<ConsenusPoseFactor4D, 4, 4>(
            new ConsenusPoseFactor4D(ref_pose.pos(), ref_pose.yaw(), rho_T, rho_theta));
    }

};
}