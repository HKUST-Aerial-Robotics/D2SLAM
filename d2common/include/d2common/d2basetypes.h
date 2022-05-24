#pragma once
#include <stdint.h>
#include <Eigen/Eigen>

#define POSE_SIZE 7
#define POSE_EFF_SIZE 6
#define FRAME_SPDBIAS_SIZE 9
#define TD_SIZE 1
#define INV_DEP_SIZE 1
#define POS_SIZE 3
using namespace Eigen;

namespace D2Common {
typedef int64_t FrameIdType;
typedef int64_t LandmarkIdType;
typedef int32_t CamIdType;
typedef double state_type;
typedef SparseMatrix<state_type> SparseMat;
};