#include <d2vins/d2vins_types.hpp>
#include "factors/integration_base.h"

namespace D2VINS {
VINSFrame::VINSFrame(const D2FrontEnd::VisualImageDescArray & frame, const IMUBuffer & buf, const VINSFrame & prev_frame):
    stamp(frame.stamp),
    frame_id(frame.frame_id),
    drone_id(frame.drone_id),
    is_keyframe(frame.is_keyframe),
    odom(frame.stamp), Ba(prev_frame.Ba), Bg(prev_frame.Bg),
    prev_frame_id(prev_frame.frame_id) {
    pre_integrations = new IntegrationBase(buf, Ba, Bg);
}

VINSFrame::VINSFrame(const D2FrontEnd::VisualImageDescArray & frame, const Vector3d & _Ba, const Vector3d & _Bg):
    stamp(frame.stamp),
    frame_id(frame.frame_id),
    odom(frame.stamp), Ba(_Ba), Bg(_Bg) {}

std::string VINSFrame::toStr() {
    char buf[256] = {0};
    sprintf(buf, "VINSFrame %ld@%d Odom: %s", frame_id, drone_id, odom.toStr().c_str());
    return std::string(buf);
}

void VINSFrame::toVector(state_type * _pose, state_type * _spd_bias) const {
    odom.pose().to_vector(_pose);
    _spd_bias[0] = odom.vel().x();
    _spd_bias[1] = odom.vel().y();
    _spd_bias[2] = odom.vel().z();

    _spd_bias[3] = Ba.x();
    _spd_bias[4] = Ba.y();
    _spd_bias[5] = Ba.z();
    
    _spd_bias[6] = Bg.x();
    _spd_bias[7] = Bg.y();
    _spd_bias[8] = Bg.z();
}

void VINSFrame::fromVector(state_type * _pose, state_type * _spd_bias) {
    odom.pose().from_vector(_pose);
    
    odom.vel().x() = _spd_bias[0];
    odom.vel().y() = _spd_bias[1];
    odom.vel().z() = _spd_bias[2];
    
    Ba.x() = _spd_bias[3];
    Ba.y() = _spd_bias[4];
    Ba.z() = _spd_bias[5];

    Bg.x() = _spd_bias[6];
    Bg.y() = _spd_bias[7];
    Bg.z() = _spd_bias[8];
}
}