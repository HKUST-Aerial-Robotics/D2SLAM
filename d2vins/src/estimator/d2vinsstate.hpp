#include "landmark_manager.hpp"
#include "d2vins/d2vins_types.hpp"

using namespace Eigen;
namespace D2VINS {
struct VINSFrame {
    double stamp;
    int frame_id;
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
};

class D2EstimatorState {
protected:
    std::vector<VINSFrame> sld_win;
    std::vector<Swarm::Pose> extrinsic; //extrinsic of cameras
public:
    double td = 0.0; //estimated td;
    void addFrame(const VINSFrame & frame, bool is_keyframe) {

    }
};
}