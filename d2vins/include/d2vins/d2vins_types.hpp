#pragma once
#include "d2frontend/d2frontend_types.h"
#include "d2vins_params.hpp"
#include "utils.hpp"
#include <swarm_msgs/Odometry.h>
#include "d2imu.hpp"

using D2FrontEnd::LandmarkPerId;
using D2FrontEnd::LandmarkPerFrame;
using D2FrontEnd::FrameIdType;
using D2FrontEnd::CamIdType;
using D2FrontEnd::LandmarkIdType;
using D2FrontEnd::LandmarkFlag;
using D2FrontEnd::LandmarkSolverFlag;
using D2FrontEnd::VisualImageDescArray;

namespace D2VINS {
class IntegrationBase;

typedef Eigen::SparseMatrix<state_type> SparseMat;

struct VINSFrame {
    double stamp = 0;
    FrameIdType frame_id = -1;
    int drone_id = -1;
    bool is_keyframe = false;
    Swarm::Odometry odom;
    Vector3d Ba; // bias of acc
    Vector3d Bg; //bias of gyro
    FrameIdType prev_frame_id = -1;
    IntegrationBase * pre_integrations = nullptr;
    VINSFrame():Ba(0., 0., 0.), Bg(0., 0., 0.)
    {}
    
    VINSFrame(const VisualImageDescArray & frame, const IMUBuffer & buf, const VINSFrame & prev_frame);
    
    VINSFrame(const VisualImageDescArray & frame, const Vector3d & _Ba, const Vector3d & _Bg);

    std::string toStr();

    void toVector(state_type * _pose, state_type * _spd_bias) const;

    void fromVector(state_type * _pose, state_type * _spd_bias);
};


    
}