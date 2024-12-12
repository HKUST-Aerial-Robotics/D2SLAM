#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../d2vins_params.hpp"

namespace D2VINS {
class OneFrameDepth {
  public:
    OneFrameDepth(double depth):
        _inv_dep(1/depth) {
        sqrt_inf = params->depth_sqrt_inf;
    }
    template<typename T>
    bool operator()(const T* const x, T *_residual) const {
        _residual[0] = (x[0] - _inv_dep)*sqrt_inf;
        return true;
    }
    double _inv_dep;
    double sqrt_inf = 10.0;

    static std::shared_ptr<ceres::CostFunction> Create(double depth) {
        return std::shared_ptr<ceres::CostFunction>(
            new ceres::AutoDiffCostFunction<OneFrameDepth, 1, 1>(
                new OneFrameDepth(depth)),
            [](ceres::CostFunction* ptr) { delete ptr; });
    }
};
}