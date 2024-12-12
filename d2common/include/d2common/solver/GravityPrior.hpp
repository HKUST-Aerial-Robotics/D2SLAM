#pragma once
#include <ceres/ceres.h>
#include <swarm_msgs/Pose.h>
#include "BaseParamResInfo.hpp"
#include "../utils.hpp"

namespace D2Common {
class GravityPriorPerturbAD { //Perturb Relpose.
public:
    template <typename T>
    bool operator()(const T *const pose_a_ptr,
                    T *residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> theta_a(pose_a_ptr + 3);
        Eigen::Quaternion<T> q_a = qa0.template cast<T>()*Utility::quatfromRotationVector(theta_a);
        Eigen::Matrix<T, 3, 3> R_a = q_a.toRotationMatrix();
        Eigen::Map<Eigen::Matrix<T, 1, 3>> residuals(residuals_ptr);
        //The last row is direction of gravity.
        residuals = R_a.bottomRows(1) - ego_motion_R_3.template cast<T>();
        // Scale the residuals by the measurement uncertainty.
        residuals.applyOnTheRight(sqrt_information_.template cast<T>());
        return true;
    }

    GravityPriorPerturbAD(const Swarm::Pose & _ego_motion_pose,
                         const Eigen::Matrix3d & _sqrt_information, 
                         const Eigen::Quaterniond & q0)
        : ego_motion_R(_ego_motion_pose.R()), sqrt_information_(_sqrt_information), qa0(q0) {
        ego_motion_R_3 = ego_motion_R.block<1, 3>(2, 0);
    }

    static std::shared_ptr<ceres::CostFunction> Create(const Swarm::Pose & _ego_motion_pose,
                         const Eigen::Matrix3d & _sqrt_information, 
                         const Eigen::Quaterniond & q0) {
        return std::make_shared<ceres::AutoDiffCostFunction<GravityPriorPerturbAD, 3, 6>>(
            new GravityPriorPerturbAD(_ego_motion_pose, _sqrt_information, q0));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // The measurement for the position of B relative to A in the A frame.
    const Eigen::Matrix3d ego_motion_R;
    Eigen::RowVector3d ego_motion_R_3;
    // The square root of the measurement information matrix.
    const Eigen::Matrix3d sqrt_information_;
    const Eigen::Quaterniond qa0;
};

class GravityPriorResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    bool is_perturb = false;
    GravityPriorResInfo():ResidualInfo(ResidualType::GravityPriorResidual) {}
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(createFramePose(state, frame_ida, is_perturb));
        return params_list;
    }
    static std::shared_ptr<GravityPriorResInfo> create(
            const std::shared_ptr<ceres::CostFunction>& cost_function,
            const std::shared_ptr<ceres::LossFunction>& loss_function, 
            FrameIdType frame_ida, bool is_perturb=true) {
        auto info = std::make_shared<GravityPriorResInfo>();
        info->frame_ida = frame_ida;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        info->is_perturb = is_perturb;
        return info;
    }
};
}