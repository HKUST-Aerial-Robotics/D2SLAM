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
#define ROTMAT_SIZE 9
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

enum ESTIMATION_MODE {
    SINGLE_DRONE_MODE, //Not accept remote frame
    SOLVE_ALL_MODE, //Each drone solve all the information
    DISTRIBUTED_CAMERA_CONSENUS, //Distributed camera consensus
    SERVER_MODE //In this mode receive all remote and solve them
};

enum CameraConfig{
    STEREO_PINHOLE = 0,
    STEREO_FISHEYE = 1,
    PINHOLE_DEPTH = 2,
    FOURCORNER_FISHEYE = 3,
    MONOCULAR = 4,
};

enum PGO_POSE_DOF {
    PGO_POSE_4D,
    PGO_POSE_6D
};

};
