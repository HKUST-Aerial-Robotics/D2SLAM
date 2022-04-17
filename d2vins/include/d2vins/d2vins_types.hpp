#pragma once
#include "d2frontend/d2frontend_types.h"
#include "d2vins_params.hpp"
#include "utils.hpp"
#include <swarm_msgs/Odometry.h>
#include "d2imu.hpp"

namespace D2VINS {
struct VINSFrame {
    double stamp;
    int frame_id;
    int drone_id;
    bool is_keyframe = false;
    Swarm::Odometry odom;
    Vector3d Ba; // bias of acc
    Vector3d Bg; //bias of gyro

    VINSFrame():Ba(0., 0., 0.), Bg(0., 0., 0.)
    {}
    
    VINSFrame(const D2Frontend::VisualImageDescArray & frame):
        stamp(frame.stamp),
        frame_id(frame.frame_id),
        odom(frame.stamp), Ba(0., 0., 0.), Bg(0., 0., 0.) {
    }

    std::string toStr() {
        char buf[256] = {0};
        sprintf(buf, "VINSFrame %d@%d Odom: %s", frame_id, drone_id, odom.toStr().c_str());
        return std::string(buf);
    }
};


    
}