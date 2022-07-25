#pragma once
#include <ceres/ceres.h>
#include <swarm_msgs/Pose.h>
#include "BaseParamResInfo.hpp"
#include "../utils.hpp"

namespace D2Common {
class RelPoseFactor {
public:
    RelPoseFactor(const Swarm::Pose &t_ab_measured,
                         const Eigen::Matrix6d &sqrt_information)
        : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information),
        sqrt_information_diag_(sqrt_information.diagonal()) {
            // std::cout << "sqrt_information_diag_" << sqrt_information_diag_.transpose() << std::endl;
        }

    template <typename T>
    bool operator()(const T *const pose_a_ptr,
                    const T *const pose_b_ptr,
                    T *residuals_ptr) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(pose_a_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_a(pose_a_ptr + 3);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(pose_b_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_b(pose_b_ptr + 3);

        // Compute the relative transformation between the two frames.
        Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
        Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

        // Represent the displacement between the two frames in the A frame.
        Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

        // Compute the error between the two orientation estimates.
        Eigen::Quaternion<T> delta_q =
            t_ab_measured_.att().template cast<T>() * q_ab_estimated.conjugate();

        // Compute the residuals.
        // [ position         ]   [ delta_p          ]
        // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
        residuals.template block<3, 1>(0, 0) =
            p_ab_estimated - t_ab_measured_.pos().template cast<T>();
        residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

        // Scale the residuals by the measurement uncertainty.
    #ifdef USE_INFORMATION_DIAG
        for (unsigned int i = 0; i < 6; i ++ ) {
            residuals(i) = residuals(i)*sqrt_information_diag_(i);
        }
    #else
        residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
    #endif

        
        return true;
    }

    static ceres::CostFunction *Create(
        const Swarm::Pose &t_ab_measured,
        const Eigen::Matrix6d &sqrt_information)
    {
        return new ceres::AutoDiffCostFunction<RelPoseFactor, 6, 7, 7>(
            new RelPoseFactor(t_ab_measured, sqrt_information));
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // The measurement for the position of B relative to A in the A frame.
    const Swarm::Pose t_ab_measured_;
    // The square root of the measurement information matrix.
    const Eigen::Matrix6d sqrt_information_;
    const Eigen::Matrix<double, 6, 1> sqrt_information_diag_;
};


class RelPoseFactor4D {
    Swarm::Pose relative_pose;
    Eigen::Vector3d relative_pos;
    double relative_yaw;
    Eigen::Matrix4d sqrt_inf;
public:
    RelPoseFactor4D(const Swarm::Pose & _relative_pose, const Eigen::Matrix4d & _sqrt_inf):
        relative_pose(_relative_pose), sqrt_inf(_sqrt_inf) {
        relative_pos = relative_pose.pos();
        relative_yaw = relative_pose.yaw();
    }

    RelPoseFactor4D(const Swarm::Pose & _relative_pose, const Eigen::Matrix3d & _sqrt_inf_pos, double sqrt_info_yaw):
        relative_pose(_relative_pose) {
        relative_pos = relative_pose.pos();
        relative_yaw = relative_pose.yaw();
        sqrt_inf.setZero();
        sqrt_inf.block<3, 3>(0, 0) = _sqrt_inf_pos;
        sqrt_inf(3, 3) = sqrt_info_yaw;
    }
    
    template<typename T>
    bool operator()(const T* const p_a_ptr, const T* const p_b_ptr, T *_residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_a(p_a_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_b(p_b_ptr);
        const Eigen::Matrix<T, 3, 1> _relative_pos = relative_pos.template cast<T>();
        const Eigen::Matrix<T, 4, 4> _sqrt_inf = sqrt_inf.template cast<T>();
        T relyaw_est = Utility::NormalizeAngle(p_b_ptr[3] - p_a_ptr[3]);
        Eigen::Matrix<T, 3, 1> relpos_est = Utility::yawRotMat(-p_a_ptr[3]) * (pos_b - pos_a);
        Utility::poseError4D<T>(relpos_est, relyaw_est, _relative_pos, (T)(relative_yaw) , _sqrt_inf, _residual);
        return true;
    }

    static ceres::CostFunction* Create(const Swarm::GeneralMeasurement2Drones* _loc) {
        auto loop = static_cast<const Swarm::LoopEdge*>(_loc);
        // std::cout << "Loop" << "sqrt_inf\n" << loop->get_sqrt_information_4d() << std::endl;
        return new ceres::AutoDiffCostFunction<RelPoseFactor4D, 4, 4, 4>(
            new RelPoseFactor4D(loop->relative_pose, loop->get_sqrt_information_4d()));
    }

    static ceres::CostFunction * Create(const Swarm::Pose & _relative_pose, const Eigen::Matrix3d & _sqrt_inf_pos, double sqrt_info_yaw) {
        return new ceres::AutoDiffCostFunction<RelPoseFactor4D, 4, 4, 4>(
            new RelPoseFactor4D(_relative_pose, _sqrt_inf_pos, sqrt_info_yaw));
    }
};

class RelPoseResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    bool is_4dof = false;
    RelPoseResInfo():ResidualInfo(ResidualType::RelPoseResidual) {}
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        std::vector<ParamInfo> params_list;
        if (is_4dof) {
            params_list.push_back(createFramePose4D(state, frame_ida));
            params_list.push_back(createFramePose4D(state, frame_idb));
        } else {
            params_list.push_back(createFramePose(state, frame_ida));
            params_list.push_back(createFramePose(state, frame_idb));
        }
        return params_list;
    }

    static RelPoseResInfo * create(
            ceres::CostFunction * cost_function, ceres::LossFunction * loss_function, 
            FrameIdType frame_ida, FrameIdType frame_idb, bool is_4dof=false) {
        auto * info = new RelPoseResInfo();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        info->is_4dof = is_4dof;
        return info;
    }
};
}