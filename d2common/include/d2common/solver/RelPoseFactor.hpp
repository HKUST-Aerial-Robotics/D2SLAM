#pragma once
#include <ceres/ceres.h>
#include <swarm_msgs/Pose.h>
#include "BaseParamResInfo.hpp"
#include "../utils.hpp"

namespace D2Common {
class RelPoseFactor : public ceres::SizedCostFunction<6, 7, 7> {
    Matrix3d q_sqrt_info;
    Matrix3d T_sqrt_info;
    Vector3d t_rel;
    Quaterniond q_rel;
public:
    RelPoseFactor(Swarm::Pose relative_pose, Matrix6d sqrt_info): 
        q_sqrt_info(sqrt_info.block<3,3>(3,3)), T_sqrt_info(sqrt_info.block<3,3>(0,0)) {
        t_rel = relative_pose.pos();
        q_rel = relative_pose.att();
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        Map<const Vector3d> t0(parameters[0]);
        Map<const Quaterniond> q0(parameters[0] + 3);
        Map<const Vector3d> t1(parameters[1]);
        Map<const Quaterniond> q1(parameters[1] + 3);
        Map<Vector3d> T_err(residuals);
        Map<Vector3d> theta_err(residuals + 3);
        Quaterniond q_0_inv = q0.conjugate();
        Matrix3d R_0_inv = q_0_inv.toRotationMatrix();
        Quaterniond q_01_est = q_0_inv * q1;
        Vector3d T_01_est = q_0_inv*(t1 - t0);  
        Quaterniond q_err = Utility::positify(q_rel.inverse() * q_01_est);
        theta_err = 2.0 * q_sqrt_info * q_err.vec();
        T_err = T_sqrt_info*(T_01_est - t_rel);
        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_0(jacobians[0]);
                jacobian_pose_0.setZero();
                jacobian_pose_0.block<3, 3>(0, 0) = -T_sqrt_info*R_0_inv;
                jacobian_pose_0.block<3, 3>(0, 3) = T_sqrt_info*Utility::skewSymmetric(R_0_inv*(t1 - t0)); 
                jacobian_pose_0.block<3, 3>(3, 3) = -q_sqrt_info*Utility::Qleft(q1.inverse() * q0).bottomRightCorner<3, 3>();
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_1(jacobians[1]);
                jacobian_pose_1.setZero();
                jacobian_pose_1.block<3, 3>(0, 0) = T_sqrt_info*R_0_inv;
                jacobian_pose_1.block<3, 3>(3, 3) = q_sqrt_info*Utility::Qleft(q_0_inv * q1).bottomRightCorner<3, 3>();
            }

        }
        return true;
    }
    static std::shared_ptr<ceres::CostFunction> Create(const Swarm::Pose & _relative_pose, const Eigen::Matrix6d & _sqrt_inf) {
        return std::make_shared<RelPoseFactor>(_relative_pose, _sqrt_inf);
    }
    
    static std::shared_ptr<ceres::CostFunction> Create(const Swarm::LoopEdge loop) {
        return std::make_shared<RelPoseFactor>(loop.relative_pose, loop.getSqrtInfoMat());
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // The measurement for the position of B relative to A in the A frame.
    const Swarm::Pose t_ab_measured_;
    // The square root of the measurement information matrix.
    const Eigen::Matrix6d sqrt_information_;
};

class RelPoseFactorAD { //AD stands for AutoDiff
public:
    RelPoseFactorAD(const Swarm::Pose &t_ab_measured,
                         const Eigen::Matrix6d &sqrt_information)
        : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information) {
            // std::cout << "sqrt_information_diag_" << sqrt_information_diag_.transpose() << std::endl;
        }

    template <typename T>
    bool operator()(const T *const pose_a_ptr,
                    const T *const pose_b_ptr,
                    T *residuals_ptr) const {
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

    static std::shared_ptr<ceres::CostFunction>Create(
        const Swarm::Pose &t_ab_measured,
        const Eigen::Matrix6d &sqrt_information)
    {
        return std::make_shared<ceres::AutoDiffCostFunction<RelPoseFactorAD, 6, 7, 7>>(
            new RelPoseFactorAD(t_ab_measured, sqrt_information));
    }

    static std::shared_ptr<ceres::CostFunction> Create(const Swarm::LoopEdge & loop) {
        return std::make_shared<ceres::AutoDiffCostFunction<RelPoseFactorAD, 6, 7, 7>>(
            new RelPoseFactorAD(loop.relative_pose, loop.getSqrtInfoMat()));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // The measurement for the position of B relative to A in the A frame.
    const Swarm::Pose t_ab_measured_;
    // The square root of the measurement information matrix.
    const Eigen::Matrix6d sqrt_information_;
};


class RelPoseFactorPerturbAD { //Perturb Relpose.
public:
    template <typename T>
    bool operator()(const T *const pose_a_ptr,
                    const T *const pose_b_ptr,
                    T *residuals_ptr) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(pose_a_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> theta_a(pose_a_ptr + 3);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(pose_b_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> theta_b(pose_b_ptr + 3);
        Eigen::Quaternion<T> q_a = qa0.template cast<T>()*Utility::quatfromRotationVector(theta_a);
        Eigen::Quaternion<T> q_b = qb0.template cast<T>()*Utility::quatfromRotationVector(theta_b);
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
        residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
        return true;
    }

    RelPoseFactorPerturbAD(const Swarm::Pose &t_ab_measured,
                         const Eigen::Matrix6d &sqrt_information, 
                         const Eigen::Quaterniond & q0, const Eigen::Quaterniond & q1)
        : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information), qa0(q0), qb0(q1) {
            // std::cout << "sqrt_information_diag_" << sqrt_information_diag_.transpose() << std::endl;
    }

    static std::shared_ptr<ceres::CostFunction> Create(const Swarm::LoopEdge & loop, Eigen::Quaterniond & q0, Eigen::Quaterniond & q1) {
        return std::make_shared<ceres::AutoDiffCostFunction<RelPoseFactorPerturbAD, 6, 6, 6>>(
            new RelPoseFactorPerturbAD(loop.relative_pose, loop.getSqrtInfoMat(), q0, q1));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // The measurement for the position of B relative to A in the A frame.
    const Swarm::Pose t_ab_measured_;
    // The square root of the measurement information matrix.
    const Eigen::Matrix6d sqrt_information_;
    const Eigen::Quaterniond qa0;
    const Eigen::Quaterniond qb0;
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

    static std::shared_ptr<ceres::CostFunction> Create(const Swarm::LoopEdge & loop) {
        return std::make_shared<ceres::AutoDiffCostFunction<RelPoseFactor4D, 4, 4, 4>>(
            new RelPoseFactor4D(loop.relative_pose, loop.getSqrtInfoMat4D()));
    }

    static std::shared_ptr<ceres::CostFunction> Create(const Swarm::Pose & _relative_pose, const Eigen::Matrix3d & _sqrt_inf_pos, double sqrt_info_yaw) {
        return std::make_shared<ceres::AutoDiffCostFunction<RelPoseFactor4D, 4, 4, 4>>(
            new RelPoseFactor4D(_relative_pose, _sqrt_inf_pos, sqrt_info_yaw));
    }
};

class RelRotFactor9D {
    Matrix3d R_sqrt_info;
    Matrix3d R_rel;
public:
    RelRotFactor9D(Swarm::Pose relative_pose, Matrix6d sqrt_info): 
        R_sqrt_info(sqrt_info.block<3,3>(3,3)) {
        R_rel = relative_pose.R();
    }

    template<typename T>
    bool operator() (const T* const R_a_ptr, const T* const R_b_ptr, T *residuals) const {
        Map<const Matrix<T, 3, 3, RowMajor>> Ri(R_a_ptr);
        Map<const Matrix<T, 3, 3, RowMajor>> Rj(R_b_ptr);
        Map<Matrix<T, 3, 3, RowMajor>> R_res(residuals);
        R_res = R_sqrt_info*(R_rel.transpose()*Ri.transpose() - Rj.transpose());
        return true;
    }

    static std::shared_ptr<ceres::CostFunction> Create(const Swarm::Pose & _relative_pose, const Eigen::Matrix6d & _sqrt_inf) {
        return std::make_shared<ceres::AutoDiffCostFunction<RelRotFactor9D, 9, 9, 9>>(
                new RelRotFactor9D(_relative_pose, _sqrt_inf));
    }
    
    static std::shared_ptr<ceres::CostFunction>Create(const Swarm::LoopEdge & loop) {
        return std::make_shared<ceres::AutoDiffCostFunction<RelRotFactor9D, 9, 9, 9>>(
            new RelRotFactor9D(loop.relative_pose, loop.getSqrtInfoMat()));
    }
};

class RelRot9DResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;

    RelRot9DResInfo():ResidualInfo(ResidualType::RelRotResidual) {}
    
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        std::vector<ParamInfo> params_list{createFrameRotMat(state, frame_ida), 
                createFrameRotMat(state, frame_idb)};
        return params_list;
    }
    
    static std::shared_ptr<RelRot9DResInfo> create(const std::shared_ptr<ceres::CostFunction>& cost_function,
            const std::shared_ptr<ceres::LossFunction> loss_function, 
            FrameIdType frame_ida, FrameIdType frame_idb) {
        auto info = std::make_shared<RelRot9DResInfo>();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        return info;
    }
};

class RelPoseResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    bool is_4dof = false;
    bool is_perturb = false;
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
            params_list.push_back(createFramePose(state, frame_ida, is_perturb));
            params_list.push_back(createFramePose(state, frame_idb, is_perturb));
        }
        return params_list;
    }
    static std::shared_ptr<RelPoseResInfo> create(
            const std::shared_ptr<ceres::CostFunction>& cost_function, const std::shared_ptr<ceres::LossFunction>& loss_function, 
            FrameIdType frame_ida, FrameIdType frame_idb, bool is_4dof=false, bool is_perturb=false) {
        auto info = std::make_shared<RelPoseResInfo>();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        info->is_4dof = is_4dof;
        info->is_perturb = is_perturb;
        return info;
    }
};
}