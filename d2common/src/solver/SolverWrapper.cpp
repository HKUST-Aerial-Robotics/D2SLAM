#include <d2common/solver/BaseParamResInfo.hpp>
#include <d2common/solver/SolverWrapper.hpp>

namespace D2Common {

SolverWrapper::SolverWrapper(D2State * _state): state(_state) {
    problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem = new ceres::Problem(problem_options);
}
void SolverWrapper::addResidual(const std::shared_ptr<ResidualInfo>& residual_info) {
    residuals.push_back(residual_info);
}

ceres::Problem & SolverWrapper::getProblem() {
    return *problem;
}
void SolverWrapper::reset() {
    delete problem;
    problem = new ceres::Problem(problem_options);
    residuals.clear();
}

void CeresSolver::addResidual(const std::shared_ptr<ResidualInfo>& residual_info) {
  auto pointers = residual_info->paramsPointerList(state);
  // printf("Add residual info %d", residual_info->residual_type);
  problem->AddResidualBlock(CheckGetPtr(residual_info->cost_function),
                            residual_info->loss_function.get(), pointers); // loss_function maybe nullptr
  SolverWrapper::addResidual(residual_info);
}

SolverReport CeresSolver::solve() {
  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);
  SolverReport report;
  report.total_iterations =
      summary.num_successful_steps + summary.num_unsuccessful_steps;
  report.total_time = summary.total_time_in_seconds;
  report.initial_cost = summary.initial_cost;
  report.final_cost = summary.final_cost;
  report.summary = summary;
  // std::cout << summary.FullReport() << std::endl;
  return report;
}

}  // namespace D2Common
