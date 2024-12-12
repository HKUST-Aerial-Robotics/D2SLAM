#pragma once
#include <d2common/d2basetypes.h>
#include "../d2state.hpp"
#include <ceres/ceres.h>

namespace D2Common {
enum ParamsType {
    POSE = 0,
    POSE_4D,
    POSE_PERTURB_6D,
    ROTMAT,
    REL_COOR, //Relative cooridinate frame pose (P_w_i_k)
    SPEED_BIAS,
    EXTRINSIC,
    TD,
    LANDMARK,
};

inline bool IsSE3(ParamsType type) {
    return type == POSE || type == REL_COOR || type == EXTRINSIC;
}

inline bool IsPose4D(ParamsType type) {
    return type == POSE_4D;
}

struct ParamInfo {
    StatePtr pointer = nullptr;
    Matrix<state_type, -1, 1> data_copied;
    int index = -1;
    int size = 0;
    int eff_size = 0; //This is size on tangent space.
    bool is_remove = false;
    ParamsType type;
    FrameIdType id;
    ParamInfo() {}
    state_type * getPointer() {
        return CheckGetPtr(pointer);
    }
};

enum ResidualType {
    NONE, // 0
    IMUResidual, // 1
    LandmarkTwoFrameOneCamResidual, // 2
    LandmarkTwoFrameTwoCamResidual, // 3
    LandmarkTwoDroneTwoCamResidual, // 4
    LandmarkOneFrameTwoCamResidual, // 5
    PriorResidual, // 6
    DepthResidual, // 7
    RelPoseResidual, //8
    RelRotResidual, //9
    GravityPriorResidual //10
};

class ResidualInfo {
public:
    ResidualType residual_type;
    std::shared_ptr<ceres::CostFunction> cost_function = nullptr;
    std::shared_ptr<ceres::LossFunction> loss_function = nullptr;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians; //Jacobian of each parameter blocks
    VectorXd residuals;
    ResidualInfo(ResidualType type) : residual_type(type) {} 
    virtual void Evaluate(D2State * state);
    virtual void Evaluate(const std::vector<ParamInfo> & param_infos, bool use_copied=false);
    virtual bool relavant(const std::set<FrameIdType> & frame_id) const = 0;
    virtual std::vector<ParamInfo> paramsList(D2State * state) const = 0;
    virtual std::vector<state_type*> paramsPointerList(D2State * state) const {
        std::vector<state_type*> params;
        for (auto info : paramsList(state)) {
            params.push_back(CheckGetPtr(info.pointer));
        }
        return params;
    }
    int residualSize() const {
        return cost_function->num_residuals();
    }
    virtual ~ResidualInfo(){}
};

ParamInfo createFramePose(D2State * state, FrameIdType id, bool is_perturb=false);
ParamInfo createFrameRotMat(D2State * state, FrameIdType id);
ParamInfo createFramePose4D(D2State * state, FrameIdType id);

}