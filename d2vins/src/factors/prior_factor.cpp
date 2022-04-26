#include "prior_factor.h"
#include "../estimator/marginalize.hpp"

namespace D2VINS {

MatrixXd toJacRes(const SparseMat & A, VectorXd & b);

PriorFactor::PriorFactor(std::vector<ParamInfo> _keep_params_list, const SparseMat & A, const VectorXd & b): 
    keep_params_list(_keep_params_list)
{
    // TicToc tic_j;
    linearized_res = b;
    linearized_jac = toJacRes(A, linearized_res);
    // printf("[D2VINS::Marginalizer] linearized_jac time cost %.3fms\n", tic_j.toc());
    keep_param_size = keep_params_list.size();
}

bool PriorFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::VectorXd dx(keep_param_size);
    for (int i = 0; i < keep_param_size; i++)
    {
        auto & info = keep_params_list[i];
        int size = info.eff_size;
        int idx = info.index;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(info.pointer, size);
        if (info.type != POSE && info.type != EXTRINSIC)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, keep_param_size) = linearized_res + linearized_jac * dx;
    if (jacobians)
    {

        for (int i = 0; i < keep_param_size; i++)
        {
            if (jacobians[i])
            {
                auto & info = keep_params_list[i];
                int size = info.eff_size;
                int idx = info.index;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], keep_param_size, size);
                jacobian.setZero();
                jacobian.leftCols(size) = linearized_jac.middleCols(idx, size);
            }
        }
    }
    return true;
}

std::vector<state_type*> PriorFactor::getKeepParamsPointers() const {
    std::vector<state_type *> pointers;
    for (auto & info : keep_params_list)
    {
        pointers.push_back(info.pointer);
    }
    return pointers;
}

std::vector<ParamInfo> PriorFactor::getKeepParams() const {
    return keep_params_list;
}

int PriorFactor::getParamSize() const {
    int size = 0;
    for (auto & info : keep_params_list)
    {
        size += info.size;
    }
    return size;
}


class MyLLT : public Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::NaturalOrdering<int>> {
public:
    MatrixXd matrixLDense() const {
        eigen_assert(Base::m_factorizationIsOk && "Simplicial LLT not factorized");
        return MatrixXd(Base::m_matrix);
    }
    void solveLb(VectorXd & b) {
      Traits::getL(Base::m_matrix).solveInPlace(b);
    }
};

MatrixXd toJacRes(const SparseMat & A, VectorXd & b) {
    // Ideally we should use sparse MyLLT for this, but it has some unknown bug. So we use a dense version.
    // auto ret = A.toDense();
    // MyLLT solver;
    // solver.compute(A);
    // assert(solver.info() == Eigen::Success && "LLT failed");
    // auto L = solver.matrixLDense();
    // printf("b rows %d cols %d L rows %d cols %d\n", b.rows(), b.cols(), L.rows(), L.cols());
    // solver.solveLb(b);
    // fflush(stdout);
    auto Adense = A.toDense();
    LLT<MatrixXd> llt(Adense);
    llt.matrixL().solveInPlace(b);
    return llt.matrixL();
}

}