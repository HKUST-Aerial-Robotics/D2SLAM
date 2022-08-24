#include <d2common/solver/SolverWrapper.hpp>
#include <d2common/solver/BaseParamResInfo.hpp>

namespace D2Common {
void CeresSolver::addResidual(ResidualInfo*residual_info) {
    auto pointers = residual_info->paramsPointerList(state);
    // printf("Add residual info %d", residual_info->residual_type);
    problem->AddResidualBlock(residual_info->cost_function,
                             residual_info->loss_function,
                             pointers);
    SolverWrapper::addResidual(residual_info);
}

SolverReport CeresSolver::solve() {
    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
    SolverReport report;
    report.total_iterations = summary.num_successful_steps + summary.num_unsuccessful_steps;
    report.total_time = summary.total_time_in_seconds;
    report.initial_cost = summary.initial_cost;
    report.final_cost = summary.final_cost;
    report.summary = summary;
    // std::cout << summary.FullReport() << std::endl;
    return report;
}

}