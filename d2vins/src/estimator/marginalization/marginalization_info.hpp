#pragma once
#include <d2common/d2vinsframe.h>
#include <ceres/ceres.h>
#include "../d2vinsstate.hpp"
#include "../../d2vins_params.hpp"

namespace D2VINS {
class PriorFactor;
enum ResidualType {
    NONE, // 0
    IMUResidual, // 1
    LandmarkTwoFrameOneCamResidual, // 2
    LandmarkTwoFrameTwoCamResidual, // 3
    LandmarkTwoDroneTwoCamResidual, // 4
    LandmarkOneFrameTwoCamResidual, // 5
    PriorResidual, // 7
    DepthResidual // 8
};

enum ParamsType {
    POSE = 0,
    REL_COOR, //Relative cooridinate frame pose (P_w_i_k)
    SPEED_BIAS,
    EXTRINSIC,
    TD,
    LANDMARK,
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
    
    ParamInfo paramInfoRelativeCoor(D2EstimatorState * state, int drone_id) const {
        ParamInfo info;
        info.pointer = state->getPwikState(drone_id);
        info.index = -1;
        info.size = POSE_SIZE;
        info.eff_size = POSE_EFF_SIZE;
        info.type = POSE;
        info.id = drone_id;
        info.data_copied = new state_type[info.size];
        memcpy(info.data_copied, info.pointer, sizeof(state_type) * info.size);
        return info;
    }

    ParamInfo paramInfoExtrinsic(D2EstimatorState * state, int camera_id) const {
        ParamInfo info;
        info.pointer = state->getExtrinsicState(camera_id);
        info.index = -1;
        info.size = POSE_SIZE;
        info.eff_size = POSE_EFF_SIZE;
        info.type = EXTRINSIC;
        info.id = camera_id;
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

    ParamInfo paramInfoTd(D2EstimatorState * state, int camera_id) const {
        ParamInfo info;
        info.pointer = state->getTdState(camera_id);
        info.index = -1;
        info.size = TD_SIZE;
        info.eff_size = TD_SIZE;
        info.type = TD;
        info.id = camera_id;
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
    virtual void Evaluate(D2EstimatorState * state);
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
    int camera_id;
    bool enable_depth_mea = false;
    LandmarkTwoFrameOneCamResInfo():ResidualInfo(ResidualType::LandmarkTwoFrameOneCamResidual) {}
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        if (params->remove_base_when_margin_remote == 0) {
            //In this case, only the base frame is considered in margin.
            return frame_id.find(frame_ida) != frame_id.end();
        }
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoFramePose(state, frame_ida));
        params_list.push_back(paramInfoFramePose(state, frame_idb));
        params_list.push_back(paramInfoExtrinsic(state, camera_id));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        params_list.push_back(paramInfoTd(state, camera_id));
        return params_list;
    }

    static LandmarkTwoFrameOneCamResInfo * create(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id, bool enable_depth_mea) {
            auto * info = new LandmarkTwoFrameOneCamResInfo();
            info->frame_ida = frame_ida;
            info->frame_idb = frame_idb;
            info->landmark_id = landmark_id;
            info->camera_id = camera_id;
            info->cost_function = cost_function;
            info->loss_function = loss_function;
            info->enable_depth_mea = enable_depth_mea;
            return info;
    }
};

class LandmarkTwoFrameTwoCamResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    LandmarkIdType landmark_id;
    int camera_id_a;
    int camera_id_b;
    LandmarkTwoFrameTwoCamResInfo():ResidualInfo(ResidualType::LandmarkTwoFrameTwoCamResidual) {}
    // virtual void Evaluate(D2EstimatorState * state) override;
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        if (params->remove_base_when_margin_remote == 0) {
            return frame_id.find(frame_ida) != frame_id.end();
        }
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoFramePose(state, frame_ida));
        params_list.push_back(paramInfoFramePose(state, frame_idb));
        params_list.push_back(paramInfoExtrinsic(state, camera_id_a));
        params_list.push_back(paramInfoExtrinsic(state, camera_id_b));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        params_list.push_back(paramInfoTd(state, camera_id_a));
        return params_list;
    }
    static LandmarkTwoFrameTwoCamResInfo * create(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id_a, int camera_id_b) {
        auto * info = new LandmarkTwoFrameTwoCamResInfo();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->landmark_id = landmark_id;
        info->camera_id_a = camera_id_a;
        info->camera_id_b = camera_id_b;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        return info;
    }
};

class LandmarkTwoDroneTwoCamResInfo : public ResidualInfo {
public:
    int drone_ida;
    int drone_idb;
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    LandmarkIdType landmark_id;
    int camera_id_a;
    int camera_id_b;

    LandmarkTwoDroneTwoCamResInfo(): ResidualInfo(ResidualType::LandmarkTwoDroneTwoCamResidual) {}
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        if (params->remove_base_when_margin_remote == 0) {
            return frame_id.find(frame_ida) != frame_id.end();
        }
        return frame_id.find(frame_ida) != frame_id.end() || frame_id.find(frame_idb) != frame_id.end();
    }

    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoFramePose(state, frame_ida));
        params_list.push_back(paramInfoFramePose(state, frame_idb));
        params_list.push_back(paramInfoExtrinsic(state, camera_id_a));
        params_list.push_back(paramInfoExtrinsic(state, camera_id_b));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        params_list.push_back(paramInfoTd(state, camera_id_a));
        params_list.push_back(paramInfoRelativeCoor(state, drone_ida));
        params_list.push_back(paramInfoRelativeCoor(state, drone_idb));
        return params_list;
    }

    static LandmarkTwoDroneTwoCamResInfo * create(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id_a, int camera_id_b, int drone_ida, int drone_idb) {
        auto * info = new LandmarkTwoDroneTwoCamResInfo();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->landmark_id = landmark_id;
        info->camera_id_a = camera_id_a;
        info->camera_id_b = camera_id_b;
        info->drone_ida = drone_ida;
        info->drone_idb = drone_idb;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        return info;
    }
};

class LandmarkOneFrameTwoCamResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    LandmarkIdType landmark_id;
    int camera_id_a;
    int camera_id_b;
    LandmarkOneFrameTwoCamResInfo():ResidualInfo(ResidualType::LandmarkOneFrameTwoCamResidual) {}
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(frame_ida) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list;
        params_list.push_back(paramInfoExtrinsic(state, camera_id_a));
        params_list.push_back(paramInfoExtrinsic(state, camera_id_b));
        params_list.push_back(paramInfoLandmark(state, landmark_id));
        params_list.push_back(paramInfoTd(state, camera_id_a));
        return params_list;
    }

    static LandmarkOneFrameTwoCamResInfo * create(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, LandmarkIdType landmark_id, int camera_id_a, int camera_id_b) {
        auto * info = new LandmarkOneFrameTwoCamResInfo();
        info->frame_ida = frame_ida;
        info->landmark_id = landmark_id;
        info->camera_id_a = camera_id_a;
        info->camera_id_b = camera_id_b;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        return info;
    }
};

class ImuResInfo : public ResidualInfo {
public:
    FrameIdType frame_ida;
    FrameIdType frame_idb;
    ImuResInfo():ResidualInfo(ResidualType::IMUResidual) {}
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
    static ImuResInfo * create(ceres::CostFunction * cost_function,  FrameIdType frame_ida, FrameIdType frame_idb) {
        auto * info = new ImuResInfo();
        info->frame_ida = frame_ida;
        info->frame_idb = frame_idb;
        info->cost_function = cost_function;
        info->loss_function = nullptr;
        return info;
    }
};

class DepthResInfo : public ResidualInfo {
public:
    FrameIdType base_frame_id;
    LandmarkIdType landmark_id;
    DepthResInfo():ResidualInfo(ResidualType::DepthResidual) {}
    bool relavant(const std::set<FrameIdType> & frame_id) const override {
        return frame_id.find(base_frame_id) != frame_id.end();
    }
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override {
        std::vector<ParamInfo> params_list{paramInfoLandmark(state, landmark_id)};
        return params_list;
    }
    static DepthResInfo * create(ceres::CostFunction * cost_function, ceres::LossFunction * loss_function,
        FrameIdType frame_ida, LandmarkIdType landmark_id) {
        auto * info = new DepthResInfo();
        info->base_frame_id = frame_ida;
        info->landmark_id = landmark_id;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        return info;
    }
};

class PriorResInfo : public ResidualInfo {
    PriorFactor * factor;
    std::set<state_type*> params_set;
public:
    PriorResInfo(PriorFactor * _factor);
    virtual std::vector<ParamInfo> paramsList(D2EstimatorState * state) const override;
    bool relavant(const std::set<FrameIdType> & frame_ids) const override;
};

}