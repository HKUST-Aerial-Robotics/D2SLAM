#include <d2common/solver/SolverWrapper.hpp>
#include <d2common/solver/BaseParamResInfo.hpp>

namespace D2Common {
void CeresSolver::addResidual(ResidualInfo*residual_info) {
    auto pointers = residual_info->paramsPointerList(state);
    // printf("Add residual info %d", residual_info->residual_type);
    problem->AddResidualBlock(residual_info->cost_function,
                             residual_info->loss_function,
                             pointers);
}

ceres::Solver::Summary CeresSolver::solve() {
    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
    return summary;
}

}