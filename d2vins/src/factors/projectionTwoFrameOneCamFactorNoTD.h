#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../d2vins_params.hpp"

// #define CHECK_RESIDUAL

namespace D2VINS {
class ProjectionTwoFrameOneCamFactorNoTD
{
    Eigen::Vector3d pts_i, pts_j;
    bool enable_check = false;
    bool enable_depth = false;
    double inv_dep_j = 1.0;
    double dep_sqrt_inf = 1.0;
public:
    ProjectionTwoFrameOneCamFactorNoTD(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) : pts_i(_pts_i), pts_j(_pts_j) {
    }

    ProjectionTwoFrameOneCamFactorNoTD(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j, double depth_j): 
        pts_i(_pts_i), pts_j(_pts_j), enable_depth(true), inv_dep_j(1.0/depth_j), dep_sqrt_inf(params->depth_sqrt_inf) {
    }

    void test(const double* const pose_i, const double* const pose_j, const double* const pose_ext, const double* const inv_dep) {
        enable_check = true;
        double residuals[2] = {0};
        this->operator()<double>(pose_i, pose_j, pose_ext, inv_dep, residuals);
    }

    template <typename T>
    bool operator()(const T* const pose_i, const T* const pose_j, const T* const pose_ext, const T* const inv_dep, T *residuals) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pi(pose_i);
        Eigen::Map<const Eigen::Quaternion<T>> Qi(pose_i+3);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Pj(pose_j);
        Eigen::Map<const Eigen::Quaternion<T>> Qj(pose_j+3);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> tic(pose_ext);
        Eigen::Map<const Eigen::Quaternion<T>> qic(pose_ext+3);

        auto inv_dep_i = inv_dep[0];

        Eigen::Matrix<T, 3, 1> pts_camera_i = pts_i / inv_dep_i;
        Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Matrix<T, 3, 1> pts_w = Qi * pts_imu_i + Pi;
        Eigen::Matrix<T, 3, 1> pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Matrix<T, 3, 1> pts_camera_j = qic.inverse() * (pts_imu_j - tic);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(residuals);

#ifdef UNIT_SPHERE_ERROR
        residual = tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
        auto est_inv_dep_j = 1.0/pts_camera_j.z();
        residual.x() = (pts_camera_j.x() * est_inv_dep_j) - pts_j.x();
        residual.y() = (pts_camera_j.y() * est_inv_dep_j) - pts_j.y();
#endif
#ifdef CHECK_RESIDUAL
        if (enable_check) {
            auto tmp  = qic * pts_camera_i + tic;
            printf("Qi:         : %.2f, %.2f, %.2f, %.2f\n", Qi.w(), Qi.x(), Qi.y(), Qi.z());
            printf("Pi:         : %.2f, %.2f, %.2f\n", Pi.x(), Pi.y(), Pi.z());
            printf("Qj:         : %.2f, %.2f, %.2f, %.2f\n", Qj.w(), Qj.x(), Qj.y(), Qj.z());
            printf("Pj:         : %.2f, %.2f, %.2f\n", Pj.x(), Pj.y(), Pj.z());
            printf("qic:        : %.2f, %.2f, %.2f, %.2f\n", qic.w(), qic.x(), qic.y(), qic.z());
            printf("tic:        : %.2f, %.2f, %.2f\n", tic.x(), tic.y(), tic.z());
            printf("pts_i       : %.2f, %.2f, %.2f\n", pts_i(0), pts_i(1), pts_i(2));
            printf("pts_j       : %.2f, %.2f, %.2f\n", pts_j(0), pts_j(1), pts_j(2));
            printf("inv_dep_i   : %.2f\n", inv_dep_i);
            printf("inv_dep_j   : %.2f\n", inv_dep_j);
            printf("est_inv_dep_j %.2f\n", est_inv_dep_j);
            printf("pts_camera_i: %.2f, %.2f, %.2f\n", pts_camera_i(0), pts_camera_i(1), pts_camera_i(2));
            printf("tmp         : %.2f, %.2f, %.2f\n", tmp(0), tmp(1), tmp(2));
            printf("pts_imu_i   : %.2f, %.2f, %.2f\n", pts_imu_i(0), pts_imu_i(1), pts_imu_i(2));
            printf("pts_w       : %.2f, %.2f, %.2f\n", pts_w(0), pts_w(1), pts_w(2));
            printf("pts_camera_j: %.2f, %.2f, %.2f\n", pts_camera_j(0), pts_camera_j(1), pts_camera_j(2));
            printf("resid_n_inf : %.2f, %.2f\n", residual(0), residual(1));
        }
#endif
        residual = ProjectionTwoFrameOneCamFactor::sqrt_info * residual;
        if (enable_depth) {
            residuals[2] = (est_inv_dep_j - inv_dep_j) * dep_sqrt_inf;
        }
#ifdef CHECK_RESIDUAL
        if (enable_check) {
            printf("resididual  : %.2f, %.2f\n", residual(0), residual(1));
        }
#endif
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) {
        return (new ceres::AutoDiffCostFunction<ProjectionTwoFrameOneCamFactorNoTD, 2, 7, 7, 7, 1>(
            new ProjectionTwoFrameOneCamFactorNoTD(_pts_i, _pts_j)));
    }

    
    static ceres::CostFunction *Create(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j, double depth_j) {
        return (new ceres::AutoDiffCostFunction<ProjectionTwoFrameOneCamFactorNoTD, 3, 7, 7, 7, 1>(
            new ProjectionTwoFrameOneCamFactorNoTD(_pts_i, _pts_j, depth_j)));
    }
};
}