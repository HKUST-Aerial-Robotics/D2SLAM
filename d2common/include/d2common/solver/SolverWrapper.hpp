#pragma once

#include <iostream>
#include <ceres/ceres.h>
#include <d2common/d2state.hpp>
#include <d2common/solver/BaseParamResInfo.hpp>
#include "spdlog/spdlog.h"
#include <functional>

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
    ceres::Problem::Options problem_options;
    D2State * state;
    std::vector<std::shared_ptr<ResidualInfo>> residuals;
    virtual void setStateProperties() {}
public:
    SolverWrapper(D2State * _state);
    virtual void addResidual(const std::shared_ptr<ResidualInfo>& residual_info);
    virtual SolverReport solve() = 0; // TODO: remove
    virtual SolverReport solve(std::function<void()> func_set_properties) = 0;
    ceres::Problem & getProblem();
    virtual void reset();
};

class CeresSolver : public SolverWrapper {
protected:
    ceres::Solver::Options options;
public:
    CeresSolver(D2State * _state, ceres::Solver::Options _options): 
            SolverWrapper(_state), options(_options)  {}
    //TODO: set as override
    virtual SolverReport solve() override { assert(false && "Unused");};
    virtual SolverReport solve(std::function<void()> func_set_properties) override;
};

}