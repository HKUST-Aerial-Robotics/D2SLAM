#pragma once
#include "d2frontend/d2frontend_types.h"
#include "d2vins_params.hpp"
#include "utils.hpp"
#include "d2imu.hpp"

namespace D2VINS {
struct VINSFrame {
    double stamp;
    int frame_id;
    int drone_id;
    bool is_keyframe = false;
    Swarm::Pose pose;
    Vector3d V; //Velocity
    Vector3d Ba; // bias of acc
    Vector3d Bg; //bias of gyro

    VINSFrame():V(0., 0., 0.), Ba(0., 0., 0.), Bg(0., 0., 0.)
    {}
    
    VINSFrame(const D2Frontend::VisualImageDescArray & frame):
        stamp(frame.stamp),
        frame_id(frame.frame_id),
        V(0., 0., 0.), Ba(0., 0., 0.), Bg(0., 0., 0.) {
    }

    std::string toStr() {
        char buf[256] = {0};
        sprintf(buf, "VINSFrame %d@%d Pose %s Vel %.2f %.2f %.2f", frame_id, drone_id,
            pose.tostr().c_str(), V.x(), V.y(), V.z());
        return std::string(buf);
    }
};


    
}