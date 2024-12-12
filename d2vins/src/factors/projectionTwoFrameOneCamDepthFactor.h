/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>

namespace D2VINS {
class ProjectionTwoFrameOneCamDepthFactor : public ceres::SizedCostFunction<3, 7, 7, 7, 1, 1>
{
  public:
    ProjectionTwoFrameOneCamDepthFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
    				   const Eigen::Vector3d &_velocity_i, const Eigen::Vector3d &_velocity_j,
    				   const double _td_i, const double _td_j, const double _depth_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Vector3d velocity_i, velocity_j;
    double inv_depth_j;
    double td_i, td_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix3d sqrt_info;
    static double sum_t;
};

using ProjectionTwoFrameOneCamDepthFactorPtr = std::shared_ptr<ProjectionTwoFrameOneCamDepthFactor>;

}