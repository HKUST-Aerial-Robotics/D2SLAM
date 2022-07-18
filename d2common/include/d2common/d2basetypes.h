#pragma once
#include <stdint.h>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#define POSE_SIZE 7
#define POSE4D_SIZE 4
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
typedef std::vector<cv::Point3f> Point3fVector;
typedef std::vector<cv::Point2f> Point2fVector;

enum PGO_MODE {
    PGO_MODE_NON_DIST = 0,
    PGO_MODE_DISTRIBUTED_AROCK
};

enum PGO_POSE_DOF {
    PGO_POSE_4D,
    PGO_POSE_6D
};

};