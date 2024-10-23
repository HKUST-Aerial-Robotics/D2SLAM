#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Eigen>

struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_eigen(camera_T);
        Eigen::Map<const Eigen::Quaternion<T>> q_eigen(camera_R);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_eigen(point);
        Eigen::Matrix<T, 3, 1> pt_cam = q_eigen.inverse()*(point_eigen - p_eigen);
    	residuals[0] = pt_cam.x()/pt_cam.z() - T(observed_u);
    	residuals[1] = pt_cam.y()/pt_cam.z() - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          ReprojectionError3D, 2, 4, 3, 3>(
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};