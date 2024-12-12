#pragma once
#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <d2common/d2basetypes.h>
#include <swarm_msgs/Pose.h>
#include <map>

using namespace D2Common;
namespace D2Common {
struct ParamInfo;
};
// This is devied from VINS-Mono
namespace D2VINS
{

std::pair<MatrixXd, VectorXd> toJacRes(const SparseMat & A, const VectorXd & b);
std::pair<MatrixXd, VectorXd> toJacRes(const MatrixXd & A, const VectorXd & b);

class PriorFactor : public ceres::CostFunction {
    std::vector<ParamInfo> keep_params_list;
    std::map<StatePtr, ParamInfo> keep_params_map;
    int keep_param_blk_num = 0;
    Eigen::MatrixXd linearized_jac;
    Eigen::VectorXd linearized_res;
    int keep_eff_param_dim = -1;
    void initDims(const std::vector<ParamInfo> & _keep_params_list);
public:
    template <typename MatrixType>
    PriorFactor(const std::vector<ParamInfo> & _keep_params_list, const MatrixType &A, const VectorXd &b) {
        auto ret = toJacRes(A, b);
        linearized_jac = ret.first;
        linearized_res = ret.second;

        if (hasNan()) {
            std::cout << "NaN found in Prior factor" << std::endl;
            std::cout << "A max " << MatrixXd(A).maxCoeff() << std::endl;
            std::cout << "b max " << b.maxCoeff() << std::endl;
        }
        initDims(_keep_params_list);
    }
    PriorFactor(const PriorFactor & factor):
        keep_params_list(factor.keep_params_list),
        keep_param_blk_num(factor.keep_param_blk_num),
        linearized_jac(factor.linearized_jac),
        linearized_res(factor.linearized_res),
        keep_eff_param_dim(factor.keep_eff_param_dim) {
        initDims(keep_params_list);
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    virtual std::vector<ParamInfo> getKeepParams() const;
    void removeFrame(int frame_id);
    int getEffParamsDim() const;
    bool hasNan() const;
    void moveByPose(const Swarm::Pose & delta_pose);
    void replacetoPrevLinearizedPoints(std::vector<ParamInfo> & params);
};

using PriorFactorPtr = std::shared_ptr<PriorFactor>;

}
