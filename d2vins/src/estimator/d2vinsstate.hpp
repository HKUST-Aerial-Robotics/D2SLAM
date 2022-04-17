#include "landmark_manager.hpp"
#include "d2vins/d2vins_types.hpp"

using namespace Eigen;
namespace D2VINS {

class D2EstimatorState {
protected:
    std::vector<VINSFrame> sld_win;
    std::vector<Swarm::Pose> extrinsic; //extrinsic of cameras
public:
    double td = 0.0; //estimated td;
    void addFrame(const VINSFrame & frame, bool is_keyframe) {
    }

    VINSFrame lastFrame() const {
        return sld_win.back();
    }
};
}