#include <d2common/solver/SolverWrapper.hpp>

namespace D2Common {
SolverWrapper::SolverWrapper(D2State * _state): state(_state) {
    // problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    // problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    // problem_options.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    // problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem = new ceres::Problem(problem_options);
}
void SolverWrapper::addResidual(ResidualInfo*residual_info) {
    residuals.push_back(residual_info);
}

ceres::Problem & SolverWrapper::getProblem() {
    return *problem;
}
void SolverWrapper::reset() {
    delete problem;
    problem = new ceres::Problem(problem_options);
    for (auto residual : residuals) {
        if (residual == nullptr) {
            SPDLOG_ERROR("residual is nullptr");
        }
        delete residual;
    }
    residuals.clear();
}

}
