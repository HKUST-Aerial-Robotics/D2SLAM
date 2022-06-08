#include "SolverWrapper.hpp"
#include "ResidualInfo.hpp"

namespace D2VINS {
void BaseSolverWrapper::addResidual(ResidualInfo*residual_info) {
    auto pointers = residual_info->paramsPointerList(state);
    // printf("Add residual info %d", residual_info->residual_type);
    problem.AddResidualBlock(residual_info->cost_function,
                             residual_info->loss_function,
                             pointers);
}

ceres::Solver::Summary BaseSolverWrapper::solve() {
    ceres::Solver::Summary summary;
    ceres::Solve(params->ceres_options, &problem, &summary);
    return summary;
}

}