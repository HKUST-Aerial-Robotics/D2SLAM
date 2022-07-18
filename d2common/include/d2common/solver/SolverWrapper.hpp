#pragma once

#include <iostream>
#include <ceres/ceres.h>
#include <d2common/d2state.hpp>
#include <d2common/solver/BaseParamResInfo.hpp>

namespace D2Common {
class ResidualInfo;
class D2EstimatorState;

struct SolverReport {
    int total_iterations = 0;
    double total_time = 0;
    double final_cost = 0;
    bool succ = true;
    std::string message = "";
};

class SolverWrapper {
protected:
    ceres::Problem * problem = nullptr;
    D2State * state;
    std::vector<ResidualInfo*> residuals;
    virtual void setStateProperties() {}
public:
    SolverWrapper(D2State * _state): state(_state) {
        problem = new ceres::Problem();
    }
    virtual void addResidual(ResidualInfo*residual_info) {
        residuals.push_back(residual_info);
    }
    virtual SolverReport solve() = 0;
    ceres::Problem & getProblem() {
        return *problem;
    }
    virtual void reset() {
        delete problem;
        problem = new ceres::Problem();
        for (auto residual : residuals) {
            delete residual;
        }
        residuals.clear();
    }
};

class CeresSolver : public SolverWrapper {
protected:
    ceres::Solver::Options options;
public:
    CeresSolver(D2State * _state, ceres::Solver::Options _options): 
            SolverWrapper(_state), options(_options)  {}
    virtual void addResidual(ResidualInfo*residual_info) override;
    SolverReport solve() override;
};

}