#pragma once
#include <d2vins/d2vins_types.hpp>
#include <ceres/ceres.h>
#include "d2vinsstate.hpp"

namespace D2VINS {
using std::make_pair;
enum ResidualType {
    NONE,
    IMUResidual,
    LandmarkTwoFrameOneCamResidual,
    LandmarkTwoFrameOneCamResidualTD
};

enum ParamsType {
    POSE,
    SPEED_BIAS,
    LANDMARK,
    EXTRINSIC,
    TD
};

struct ParamInfo {
    double * pointer = nullptr;
    int index = -1;
    int size = 0;
    int eff_size = 0; //This is size on tangent space.
    bool is_remove = false;
    ParamsType type;
    FrameIdType id;
    ParamInfo() {}
};

class ResidualInfo {
protected:
    ParamInfo paramInfoFramePose(D2EstimatorState * state, FrameIdType id) const {
        ParamInfo info;
        info.pointer = state->getPoseState(id);
        info.index = -1;
        info.size = POSE_SIZE;
        info.eff_size = POSE_EFF_SIZE;
        info.type = POSE;
        info.id = id;
        return info;
    }

    ParamInfo paramInfoExtrinsic(D2EstimatorState * state, int camera_id) const {
        ParamInfo info;
        info.pointer = state->getExtrinsicState(camera_id);
        info.index = -1;
        info.size = POSE_SIZE;
        info.eff_size = POSE_EFF_SIZE;
        info.type = POSE;
        info.id = camera_id;
        return info;
    }

    ParamInfo paramInfoLandmark(D2EstimatorState * state, int landmark_id) const {
        ParamInfo info;
        info.pointer = state->getLandmarkState(landmark_id);
        info.index = -1;
        if (params->landmark_param == D2VINS::D2VINSConfig::LM_INV_DEP) {
            info.size = INV_DEP_SIZE;
            info.eff_size = INV_DEP_SIZE;
        } else {
            info.size = POS_SIZE;
            info.eff_size = POS_SIZE;
        }
        info.type = LANDMARK;
        info.id = landmark_id;
        return info;
    }

    ParamInfo paramInfoSpeedBias(D2EstimatorState * state, FrameIdType id) const {
        ParamInfo info;
        info.pointer = state->getSpdBiasState(id);
        info.index = -1;
        info.size = FRAME_SPDBIAS_SIZE;
        info.eff_size = FRAME_SPDBIAS_SIZE;
        info.type = SPEED_BIAS;
        info.id = id;
        return info;
    }

    ParamInfo paramInfoTd(D2EstimatorState * state, int camera_id) const {
        ParamInfo info;
        info.pointer = state->getTdState(camera_id);
        info.index = -1;
        info.size = TD_SIZE;
        info.eff_size = TD_SIZE;
        info.type = TD;
        info.id = -1;
        return info;
    }

public:
    int parameter_size;
    ResidualType residual_type;
    ceres::CostFunction * cost_function;
    ceres::LossFunction * loss_function;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians; //Jacobian of each parameter blocks
    VectorXd residuals;
    ResidualInfo(ResidualType type) : residual_type(type) {} 
    virtual void Evaluate(std::vector<state_type*>params);
    virtual void Evaluate(D2EstimatorState * state) = 0;
    virtual bool relavant(const std::set<FrameIdType> & frame_id) const = 0;
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const = 0;
    int residualSize() const {
        return cost_function->num_residuals();
    }
    int paramSize() const {
        return parameter_size;
    }
};

class LandmarkTwoFrameOneCamResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    LandmarkIdType landmark_id;
    LandmarkTwoFrameOneCamResInfo():ResidualInfo(ResidualType::LandmarkTwoFrameOneCamResidual) {}
    int camera_id;
    virtual void Evaluate(D2EstimatorState * state) override;
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoFramePose(state, frame_ida));
        params_list.push_back(paramInfoFramePose(state, frame_idb));
        params_list.push_back(paramInfoExtrinsic(state, camera_id));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        return params_list;
    }
};

class LandmarkTwoFrameOneCamResInfoTD : public LandmarkTwoFrameOneCamResInfo {
public:
    LandmarkTwoFrameOneCamResInfoTD() {
        residual_type = ResidualType::LandmarkTwoFrameOneCamResidualTD;
    }
    virtual void Evaluate(D2EstimatorState * state) override;
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoFramePose(state, frame_ida));
        params_list.push_back(paramInfoFramePose(state, frame_idb));
        params_list.push_back(paramInfoExtrinsic(state, camera_id));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        params_list.push_back(paramInfoTd(state, camera_id));
        return params_list;
    }
};


class ImuResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    ImuResInfo():ResidualInfo(ResidualType::IMUResidual) {}
    virtual void Evaluate(D2EstimatorState * state) override;
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoFramePose(state, frame_ida));
        params_list.push_back(paramInfoSpeedBias(state, frame_ida));
        params_list.push_back(paramInfoFramePose(state, frame_idb));
        params_list.push_back(paramInfoSpeedBias(state, frame_idb));
        return params_list;
    }
};

class Marginalizer {
protected:
    D2EstimatorState * state = nullptr;
    std::vector<ResidualInfo*> residual_info_list;

    //To remove ids
    std::set<FrameIdType> remove_frame_ids;

    //Interal params
    //sorted params
    std::vector<ParamInfo> params_list; //[parameters... remove_params...] true if remove
    std::map<state_type*, ParamInfo> _params; // indx of parameters in params vector as sortd by params_list


    // void addFramePoseParams(FrameIdType frame_id);
    // void addLandmarkStateParam(LandmarkIdType frame_id);
    // void addExtrinsicParam(int camera_id);
    // void addTdStateParam(int camera_id);
    // void addParam(state_type * param, ParamsType type, FrameIdType _id, bool is_remove);
    std::pair<int, int> sortParams();
    
    VectorXd evaluate(SparseMat & J, int eff_residual_size, int eff_param_size);
    int filterResiduals();
public:
    Marginalizer(D2EstimatorState * _state): state(_state) {}
    void addLandmarkResidual(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id, bool has_td=false);
    void addImuResidual(ceres::CostFunction * cost_function, FrameIdType frame_ida, FrameIdType frame_idb);
    PriorFactor * marginalize(std::set<FrameIdType> remove_frame_ids);
};
}