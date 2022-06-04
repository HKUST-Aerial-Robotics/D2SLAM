#include <iostream>
#include "ParamInfo.hpp"
#include "ResidualInfo.hpp"
#include <ceres/ceres.h>
#include "../d2vinsstate.hpp"

namespace D2VINS {
class SolverWrapper {
protected:
    ceres::Problem problem;
    D2EstimatorState * state;
public:
    SolverWrapper(D2EstimatorState * _state): state(_state) {}
    virtual void addResidual(ResidualInfo*residual_info) = 0;
    virtual ceres::Solver::Summary solve() = 0;
    ceres::Problem & getProblem() {
        return problem;
    }
};

class BaseSolverWrapper : public SolverWrapper {
public:
    BaseSolverWrapper(D2EstimatorState * _state): SolverWrapper(_state)  {}
    virtual void addResidual(ResidualInfo*residual_info) override;
    ceres::Solver::Summary solve() override;
};

}