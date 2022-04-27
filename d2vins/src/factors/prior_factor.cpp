#include "prior_factor.h"
#include "../estimator/marginalization/marginalization.hpp"

namespace D2VINS {

MatrixXd toJacRes(const SparseMat & A, VectorXd & b);
MatrixXd toJacRes(const MatrixXd & A, VectorXd & b);
bool first_evaluate = false;
bool PriorFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // std::cout << "keep_eff_param_dim" << keep_eff_param_dim << std::endl;
    Eigen::VectorXd dx(keep_eff_param_dim);

    for (int i = 0; i < keep_param_blk_num; i++) {
        auto & info = keep_params_list[i];
        int size = info.size; //Use norminal size instead of tangent space size here.
        int idx = info.index;
        Eigen::Map<const Eigen::VectorXd> x(parameters[i], size);
        Eigen::Map<const Eigen::VectorXd> x0(info.data_copied, size);
        if (info.type != POSE && info.type != EXTRINSIC) {
            dx.segment(idx, size) = x - x0;
        } else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd> res(residuals, keep_eff_param_dim);
    res = linearized_res + linearized_jac * dx;
    
    if (jacobians) {
        for (int i = 0; i < keep_param_blk_num; i++)
        {
            if (jacobians[i])
            {
                auto & info = keep_params_list[i];
                int size = info.size; //Use norminal size instead of tangent space size here.
                int idx = info.index;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], keep_eff_param_dim, size);
                jacobian.setZero();
                jacobian.leftCols(info.eff_size) = linearized_jac.middleCols(idx, info.eff_size);
            }
        }
    }
    first_evaluate = false;
    return true;
}

std::vector<state_type*> PriorFactor::getKeepParamsPointers() const {
    std::vector<state_type *> pointers;
    // printf("prior blocks %d\n", keep_param_blk_num);
    for (auto & info : keep_params_list)
    {
        // printf("Prior info type %d id %ld\n", info.type, info.id);
        pointers.push_back(info.pointer);
    }
    return pointers;
}

std::vector<ParamInfo> PriorFactor::getKeepParams() const {
    return keep_params_list;
}

int PriorFactor::getEffParamsDim() const {
    if (keep_eff_param_dim < 0) {
        int size = 0;
        for (auto & info : keep_params_list)
        {
            size += info.eff_size;
        }
        return size;
    } else {
        return keep_eff_param_dim;
    }
}

void PriorFactor::initDims(const std::vector<ParamInfo> & _keep_params_list) {
    keep_params_list = _keep_params_list;
    keep_param_blk_num = keep_params_list.size();
    keep_eff_param_dim = getEffParamsDim();
    for (auto it : keep_params_list) {
        mutable_parameter_block_sizes()->push_back(it.size);
    }
    set_num_residuals(keep_eff_param_dim);
    first_evaluate = true;
}

MatrixXd toJacRes(const MatrixXd & A, VectorXd & b) {
    if (params->use_llt_for_decompose_A_b) {
        LLT<MatrixXd> llt(A);
        llt.matrixL().solveInPlace(b);
        b = -b;
        return llt.matrixU();
    } else {
        const double eps = 1e-8;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
        Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

        b = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
        return S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    }
}

MatrixXd toJacRes(const SparseMat & A, VectorXd & b) {
    return toJacRes(A.toDense(), b);
}

}