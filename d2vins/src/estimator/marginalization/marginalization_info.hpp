#pragma once
#include <d2vins/d2vins_types.hpp>
#include <ceres/ceres.h>
#include "../d2vinsstate.hpp"

namespace D2VINS {
class PriorFactor;
enum ResidualType {
    NONE,
    IMUResidual,
    LandmarkTwoFrameOneCamResidual,
    LandmarkTwoFrameOneCamResidualTD,
    LandmarkTwoFrameTwoCamResidualTD,
    LandmarkOneFrameTwoCamResidualTD,
    PriorResidual
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
    double * data_copied = nullptr;
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
        info.data_copied = new state_type[info.size];
        memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
        return info;
    }

    ParamInfo paramInfoExtrinsic(D2EstimatorState * state, int camera_index) const {
        ParamInfo info;
        info.pointer = state->getExtrinsicState(camera_index);
        info.index = -1;
        info.size = POSE_SIZE;
        info.eff_size = POSE_EFF_SIZE;
        info.type = EXTRINSIC;
        info.id = camera_index;
        info.data_copied = new state_type[info.size];
        memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
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
        info.data_copied = new state_type[info.size];
        memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
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
        info.data_copied = new state_type[info.size];
        memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
        return info;
    }

    ParamInfo paramInfoTd(D2EstimatorState * state, int camera_index) const {
        ParamInfo info;
        info.pointer = state->getTdState(camera_index);
        info.index = -1;
        info.size = TD_SIZE;
        info.eff_size = TD_SIZE;
        info.type = TD;
        info.id = camera_index;
        info.data_copied = new state_type[info.size];
        memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
        return info;
    }

public:
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
};

class LandmarkTwoFrameOneCamResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    LandmarkIdType landmark_id;
    int camera_index;
    LandmarkTwoFrameOneCamResInfo():ResidualInfo(ResidualType::LandmarkTwoFrameOneCamResidual) {}
    virtual void Evaluate(D2EstimatorState * state) override;
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoFramePose(state, frame_ida));
        params_list.push_back(paramInfoFramePose(state, frame_idb));
        params_list.push_back(paramInfoExtrinsic(state, camera_index));
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
        params_list.push_back(paramInfoExtrinsic(state, camera_index));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        params_list.push_back(paramInfoTd(state, camera_index));
        return params_list;
    }
};

class LandmarkTwoFrameTwoCamResInfoTD : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    LandmarkIdType landmark_id;
    int camera_index_a;
    int camera_index_b;
    LandmarkTwoFrameTwoCamResInfoTD():ResidualInfo(ResidualType::LandmarkTwoFrameTwoCamResidualTD) {}
    virtual void Evaluate(D2EstimatorState * state) override;
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoFramePose(state, frame_ida));
        params_list.push_back(paramInfoFramePose(state, frame_idb));
        params_list.push_back(paramInfoExtrinsic(state, camera_index_a));
        params_list.push_back(paramInfoExtrinsic(state, camera_index_b));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        params_list.push_back(paramInfoTd(state, camera_index_a));
        return params_list;
    }
};

class LandmarkOneFrameTwoCamResInfoTD : public ResidualInfo {
public:
    FrameIdType frame_ida;
    LandmarkIdType landmark_id;
    int camera_index_a;
    int camera_index_b;
    LandmarkOneFrameTwoCamResInfoTD():ResidualInfo(ResidualType::LandmarkOneFrameTwoCamResidualTD) {}
    virtual void Evaluate(D2EstimatorState * state) override;
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoExtrinsic(state, camera_index_a));
        params_list.push_back(paramInfoExtrinsic(state, camera_index_b));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        params_list.push_back(paramInfoTd(state, camera_index_a));
        return params_list;
    }
};

class PriorResInfo : public ResidualInfo {
    PriorFactor * factor;
    std::set<state_type*> params_set;
public:
    PriorResInfo(PriorFactor * _factor);
    virtual void Evaluate(D2EstimatorState * state) override;
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override;
    bool relavant(const std::set<FrameIdType> & frame_ids) const override;
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
}