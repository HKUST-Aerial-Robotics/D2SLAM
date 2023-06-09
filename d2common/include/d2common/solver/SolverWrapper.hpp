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
    double initial_cost = 0;
    double final_cost = 0;
    double state_changes = 0;
    bool succ = true;
    std::string message = "";
    ceres::Solver::Summary summary;
    void compose(const SolverReport & other) {
        total_iterations += other.total_iterations;
        total_time += other.total_time;
        final_cost = other.final_cost;
        succ = succ && other.succ;
        message += other.message;
        summary = other.summary;
        state_changes += other.state_changes;
    }

    double costChanges() {
        return (initial_cost - final_cost)/ initial_cost;
    }

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