#pragma once
#include <d2common/d2basetypes.h>
using namespace D2Common;

namespace D2VINS {
class D2EstimatorState;
enum ParamsType {
    POSE = 0,
    REL_COOR, //Relative cooridinate frame pose (P_w_i_k)
    SPEED_BIAS,
    EXTRINSIC,
    TD,
    LANDMARK,
};

inline bool IsSE3(ParamsType type) {
    return type == POSE || type == REL_COOR || type == EXTRINSIC;
}

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
    static ParamInfo createFramePose(D2EstimatorState * state, FrameIdType id);
    static ParamInfo createExtrinsic(D2EstimatorState * state, int camera_id);
    static ParamInfo createLandmark(D2EstimatorState * state, int landmark_id);
    static ParamInfo createSpeedBias(D2EstimatorState * state, FrameIdType id);
    static ParamInfo createTd(D2EstimatorState * state, int camera_id);
};


}