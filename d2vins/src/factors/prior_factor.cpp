#include "prior_factor.h"
#include "../estimator/marginalization/marginalization.hpp"
#include <iostream>

namespace D2VINS {

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
            printf("Param type %d index %d dx: ", info.type, idx);
            std::cout << dx.segment(idx, size).transpose();
            printf(" b: ");
            std::cout << linearized_res.segment(idx, size).transpose() << std::endl;
        } else {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
            printf("Param type %d index %d dx %f %f %f dq  %f %f %f b %f %f %f %f %f %f\n", 
                info.type, idx, dx(idx + 0), dx(idx + 1), dx(idx + 2), dx(idx + 3), dx(idx + 4), dx(idx + 5),
                linearized_res(idx + 0), linearized_res(idx + 1), linearized_res(idx + 2), linearized_res(idx + 3), linearized_res(idx + 4), linearized_res(idx + 5));
        }
    }
    Eigen::Map<Eigen::VectorXd> res(residuals, keep_eff_param_dim);
    res = linearized_res + linearized_jac * dx;
    
    if (jacobians) {
        for (int i = 0; i < keep_param_blk_num; i++) {
            if (jacobians[i]) {
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
    for (auto & info : keep_params_list) {
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
        for (auto & info : keep_params_list) {
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

std::pair<MatrixXd, VectorXd> toJacRes(const MatrixXd & A_, const VectorXd & b) {
    MatrixXd A = (A_ + A_.transpose())/2;
    if (params->use_llt_for_decompose_A_b) {
        LLT<MatrixXd> llt(A);
        VectorXd e0 = -b;
        llt.matrixL().solveInPlace(e0);
        std::cout << "l jac:" << MatrixXd(llt.matrixU()) << std::endl;
        std::cout << "l res:" << e0.transpose() << std::endl;
        return std::make_pair(llt.matrixU(), e0);
    } else {
        const double eps = 1e-8;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
        Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

        VectorXd e0 = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
        MatrixXd J_ = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();

        //Use pre-conditioned from OKVINS https://github.com/ethz-asl/okvis/blob/master/okvis_ceres/src/MarginalizationError.cpp
        // VectorXd  p = (A.diagonal().array() > eps).select(A.diagonal().cwiseSqrt(),1.0e-3);
        // VectorXd  p_inv = p.cwiseInverse();
        // SelfAdjointEigenSolver<Eigen::MatrixXd> saes(p_inv.asDiagonal() * A  * p_inv.asDiagonal() );
        // VectorXd  S_ = (saes.eigenvalues().array() > eps).select(
        //         saes.eigenvalues().array(), 0);
        // VectorXd  S_pinv_ = (saes.eigenvalues().array() > eps).select(
        //         saes.eigenvalues().array().inverse(), 0);
        // VectorXd S_sqrt_ = S_.cwiseSqrt();
        // VectorXd S_pinv_sqrt_ = S_pinv_.cwiseSqrt();

        // // assign Jacobian
        // MatrixXd J_ = (p.asDiagonal() * saes.eigenvectors() * (S_sqrt_.asDiagonal())).transpose();

        // // constant error (residual) _e0 := (-pinv(J^T) * _b):
        // Eigen::MatrixXd J_pinv_T = (S_pinv_sqrt_.asDiagonal())
        //     * saes.eigenvectors().transpose()  *p_inv.asDiagonal() ;
        // VectorXd e0 = (-J_pinv_T * b);

        Utility::writeMatrixtoFile("/home/xuhao/output/A.txt", A);
        Utility::writeMatrixtoFile("/home/xuhao/output/b.txt", b);
        Utility::writeMatrixtoFile("/home/xuhao/output/J.txt", J_);
        Utility::writeMatrixtoFile("/home/xuhao/output/e0.txt", e0);

        return std::make_pair(J_, e0);
    }
}

std::pair<MatrixXd, VectorXd> toJacRes(const SparseMat & A, const VectorXd & b) {
    return toJacRes(A.toDense(), b);
}

}