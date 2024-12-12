#pragma once
#include <d2common/d2vinsframe.h>
#include <ceres/ceres.h>
#include "../d2vins_params.hpp"
#include <d2common/solver/BaseParamResInfo.hpp>
#include "d2vinsstate.hpp"

using namespace D2Common;

namespace D2VINS {
class PriorFactor;

ParamInfo createExtrinsic(D2EstimatorState * state, int camera_id);
ParamInfo createLandmark(D2EstimatorState * state, int landmark_id, bool inv_dep_param = true);
ParamInfo createSpeedBias(D2EstimatorState * state, FrameIdType id);
ParamInfo createTd(D2EstimatorState * state, int camera_id);

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
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        std::vector<ParamInfo> params_list;
        auto _state = static_cast<D2EstimatorState*>(state);
        params_list.push_back(createFramePose(_state, frame_ida));
        params_list.push_back(createFramePose(_state, frame_idb));
        params_list.push_back(createExtrinsic(_state, camera_id));
        params_list.push_back(createLandmark(_state, landmark_id));
        params_list.push_back(createTd(_state, camera_id));
        return params_list;
    }

    static std::shared_ptr<LandmarkTwoFrameOneCamResInfo> create(std::shared_ptr<ceres::CostFunction> cost_function, std::shared_ptr<ceres::LossFunction> loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id, bool enable_depth_mea) {
            auto info = std::make_shared<LandmarkTwoFrameOneCamResInfo>();
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
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        std::vector<ParamInfo> params_list;
        auto _state = static_cast<D2EstimatorState*>(state);
        params_list.push_back(createFramePose(_state, frame_ida));
        params_list.push_back(createFramePose(_state, frame_idb));
        params_list.push_back(createExtrinsic(_state, camera_id_a));
        params_list.push_back(createExtrinsic(_state, camera_id_b));
        params_list.push_back(createLandmark(_state, landmark_id));
        params_list.push_back(createTd(_state, camera_id_a));
        return params_list;
    }
    static std::shared_ptr<LandmarkTwoFrameTwoCamResInfo> create(std::shared_ptr<ceres::CostFunction> cost_function, std::shared_ptr<ceres::LossFunction> loss_function,
        FrameIdType frame_ida, FrameIdType frame_idb, LandmarkIdType landmark_id, int camera_id_a, int camera_id_b) {
        auto info = std::make_shared<LandmarkTwoFrameTwoCamResInfo>();
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
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        std::vector<ParamInfo> params_list;
        auto _state = static_cast<D2EstimatorState*>(state);
        params_list.push_back(createExtrinsic(_state, camera_id_a));
        params_list.push_back(createExtrinsic(_state, camera_id_b));
        params_list.push_back(createLandmark(_state, landmark_id));
        params_list.push_back(createTd(_state, camera_id_a));
        return params_list;
    }

    static std::shared_ptr<LandmarkOneFrameTwoCamResInfo> create(std::shared_ptr<ceres::CostFunction> cost_function, std::shared_ptr<ceres::LossFunction> loss_function,
        FrameIdType frame_ida, LandmarkIdType landmark_id, int camera_id_a, int camera_id_b) {
        auto info = std::make_shared<LandmarkOneFrameTwoCamResInfo>();
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
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        std::vector<ParamInfo> params_list;
        auto _state = static_cast<D2EstimatorState*>(state);
        params_list.push_back(createFramePose(_state, frame_ida));
        params_list.push_back(createSpeedBias(_state, frame_ida));
        params_list.push_back(createFramePose(_state, frame_idb));
        params_list.push_back(createSpeedBias(_state, frame_idb));
        return params_list;
    }
    static std::shared_ptr<ImuResInfo> create(std::shared_ptr<ceres::CostFunction> cost_function,  FrameIdType frame_ida, FrameIdType frame_idb) {
        auto info = std::make_shared<ImuResInfo>();
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
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override {
        auto _state = static_cast<D2EstimatorState*>(state);
        std::vector<ParamInfo> params_list{createLandmark(_state, landmark_id)};
        return params_list;
    }
    static std::shared_ptr<DepthResInfo> create(std::shared_ptr<ceres::CostFunction> cost_function, std::shared_ptr<ceres::LossFunction> loss_function,
        FrameIdType frame_ida, LandmarkIdType landmark_id) {
        auto info = std::make_shared<DepthResInfo>();
        info->base_frame_id = frame_ida;
        info->landmark_id = landmark_id;
        info->cost_function = cost_function;
        info->loss_function = loss_function;
        return info;
    }
};

class PriorResInfo : public ResidualInfo {
    PriorFactorPtr factor;
public:
    PriorResInfo(const PriorFactorPtr&  _factor);
    virtual std::vector<ParamInfo> paramsList(D2State * state) const override;
    bool relavant(const std::set<FrameIdType> & frame_ids) const override;
};

}