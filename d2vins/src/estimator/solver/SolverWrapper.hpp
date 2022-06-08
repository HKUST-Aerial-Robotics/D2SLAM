#pragma once

#include <iostream>
#include <ceres/ceres.h>

namespace D2VINS {
class ResidualInfo;
class D2EstimatorState;
class SolverWrapper {
protected:
    ceres::Problem * problem = nullptr;
    D2EstimatorState * state;
public:
    SolverWrapper(D2EstimatorState * _state): state(_state) {
        problem = new ceres::Problem();
    }
    virtual void addResidual(ResidualInfo*residual_info) = 0;
    virtual ceres::Solver::Summary solve() = 0;
    ceres::Problem & getProblem() {
        return *problem;
    }
};

class BaseSolverWrapper : public SolverWrapper {
public:
    BaseSolverWrapper(D2EstimatorState * _state): SolverWrapper(_state)  {}
    virtual void addResidual(ResidualInfo*residual_info) override;
    ceres::Solver::Summary solve() override;
};

}