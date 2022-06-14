#pragma once

#include <iostream>
#include <ceres/ceres.h>
#include <d2common/d2state.hpp>

namespace D2Common {
class ResidualInfo;
class D2EstimatorState;
class SolverWrapper {
protected:
    ceres::Problem * problem = nullptr;
    D2State * state;
public:
    SolverWrapper(D2State * _state): state(_state) {
        problem = new ceres::Problem();
    }
    virtual void addResidual(ResidualInfo*residual_info) = 0;
    virtual ceres::Solver::Summary solve() = 0;
    ceres::Problem & getProblem() {
        return *problem;
    }
    virtual void reset() {
        delete problem;
        problem = new ceres::Problem();
    }
};

class CeresSolver : public SolverWrapper {
protected:
    ceres::Solver::Options options;
public:
    CeresSolver(D2State * _state, ceres::Solver::Options _options): 
            SolverWrapper(_state), options(_options)  {}
    virtual void addResidual(ResidualInfo*residual_info) override;
    ceres::Solver::Summary solve() override;
};

}