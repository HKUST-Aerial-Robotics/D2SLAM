#pragma once
#include <ceres/ceres.h>
#include <swarm_msgs/Pose.h>
#include "BaseParamResInfo.hpp"
#include "../utils.hpp"

namespace D2Common {
class RelPoseFactor : public ceres::SizedCostFunction<6, 7> {
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
            //TODO: fill in jacobians...
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_0(jacobians[0]);
                jacobian_pose_0.setZero();

                jacobian_pose_0.block<3, 3>(0, 0) = -T_sqrt_info*R_0_inv;
                jacobian_pose_0.block<3, 3>(0, 3) = T_sqrt_info*Utility::skewSymmetric(R_0_inv*(t1 - t0)); 
                jacobian_pose_0.block<3, 3>(3, 3) = -q_sqrt_info*Utility::Qleft(q1.inverse() * q0).bottomRightCorner<3, 3>();
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_1(jacobians[0]);
                jacobian_pose_1.setZero();
                jacobian_pose_1.block<3, 3>(0, 0) = T_sqrt_info*R_0_inv;
                jacobian_pose_1.block<3, 3>(3, 3) = q_sqrt_info*Utility::Qleft(q_0_inv * q1).bottomRightCorner<3, 3>();
            }

        }
        return true;
    }
};

template<typename T>
inline void yawRotateVec(T yaw, 
        const Eigen::Matrix<T, 3, 1> & vec, 
        Eigen::Matrix<T, 3, 1> & ret) {
    ret(0) = cos(yaw) * vec(0) - sin(yaw)*vec(1);
    ret(1) = sin(yaw) * vec(0) + cos(yaw)*vec(1);
    ret(2) = vec(2);
}

template<typename T>
inline Matrix<T, 3, 3> yawRotMat(T yaw) {
    Matrix<T, 3, 3> R;
    T cyaw = ceres::cos(yaw);
    T syaw = ceres::sin(yaw);
    R << cyaw, - syaw, ((T) 0), 
        syaw, cyaw, ((T) 0),
        ((T) 0), ((T) 0), ((T) 1);
    return R;
}

template <typename T>
inline T normalizeAngle(const T& angle_radians) {
    // Use ceres::floor because it is specialized for double and Jet types.
    T two_pi(2.0 * M_PI);
    return angle_radians -
            two_pi * ceres::floor((angle_radians + T(M_PI)) / two_pi);
}

template<typename T>
inline void poseError4D(const Eigen::Matrix<T, 3, 1> & posa, T yaw_a,
        const Eigen::Matrix<T, 3, 1> & posb, T yaw_b,
        const Eigen::Matrix<T, 4, 4> &_sqrt_inf_mat, 
        T *error) {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> err(error);
    err = posb - posa;
    error[3] = normalizeAngle(yaw_b - yaw_a);
    Eigen::Map<Eigen::Matrix<T, 4, 1>> err_4d(error);
    err_4d.applyOnTheLeft(_sqrt_inf_mat);
}

template<typename T>
inline void deltaPose4D(Eigen::Matrix<T, 4, 1> posea, 
        Eigen::Matrix<T, 4, 1> poseb, Eigen::Matrix<T, 4, 1> & dpose) {
    dpose(3) = normalizeAngle(poseb(3) - posea(3));
    Eigen::Matrix<T, 3, 1> tmp = poseb - posea;
    yawRotateVec(-posea(3), tmp, dpose.segment<3>(0));
}

class RelativePoseFactor4d {
    Swarm::Pose relative_pose;
    Eigen::Vector3d relative_pos;
    double relative_yaw;
    Eigen::Matrix4d sqrt_inf;
    RelativePoseFactor4d(const Swarm::Pose & _relative_pose, const Eigen::Matrix4d & _sqrt_inf):
        relative_pose(_relative_pose), sqrt_inf(_sqrt_inf) {
        relative_pos = relative_pose.pos();
        relative_yaw = relative_pose.yaw();
    }

public:
    template<typename T>
    bool operator()(const T* const p_a_ptr, const T* const p_b_ptr, T *_residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_a(p_a_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pos_b(p_b_ptr);
        const Eigen::Matrix<T, 3, 1> _relative_pos = relative_pos.template cast<T>();
        const Eigen::Matrix<T, 4, 4> _sqrt_inf = sqrt_inf.template cast<T>();
        T relyaw_est = normalizeAngle(p_b_ptr[3] - p_a_ptr[3]);
        Eigen::Matrix<T, 3, 1> relpos_est = yawRotMat(-p_a_ptr[3]) * (pos_b - pos_a);
        poseError4D(relpos_est, relyaw_est, _relative_pos, (T)(relative_yaw) , _sqrt_inf, _residual);
        return true;
    }

    static ceres::CostFunction* Create(const Swarm::GeneralMeasurement2Drones* _loc) {
        auto loop = static_cast<const Swarm::LoopEdge*>(_loc);
        // std::cout << "Loop" << "sqrt_inf\n" << loop->get_sqrt_information_4d() << std::endl;
        return new ceres::AutoDiffCostFunction<RelativePoseFactor4d, 4, 4, 4>(
            new RelativePoseFactor4d(loop->relative_pose, loop->get_sqrt_information_4d()));
    }
};

class RelPoseResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    RelPoseResInfo():ResidualInfo(ResidualType::RelPoseResidual) {}
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(createFramePose(state, frame_ida));
        params_list.push_back(createFramePose(state, frame_idb));
        return params_list;
    }
    static RelPoseResInfo * create(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function, 
            FrameIdType frame_ida, FrameIdType frame_idb) {
        auto * info = new RelPoseResInfo();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        return info;
    }
};
}