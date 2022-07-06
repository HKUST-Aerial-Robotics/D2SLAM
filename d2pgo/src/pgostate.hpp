#include <d2common/d2state.hpp>
#include <swarm_msgs/drone_trajectory.hpp>

using namespace D2Common;

namespace D2PGO {
class PGOState : public D2State {
protected:
    std::map<int, std::vector<VINSFrame*>> drone_frames;
    std::map<int, Swarm::DroneTrajectory> drone_trajs;

public:
    PGOState(int _self_id) :
        D2State(_self_id) {
    }

    void addFrame(const VINSFrame & _frame) {
        auto frame = addVINSFrame(_frame);
        if (drone_frames.find(_frame.drone_id) == drone_frames.end()) {
            drone_frames[_frame.drone_id] = std::vector<VINSFrame*>();
            drone_trajs[_frame.drone_id] = Swarm::DroneTrajectory();
        }
        drone_frames.at(_frame.drone_id).push_back(frame);
        drone_trajs[_frame.drone_id].push(frame->stamp, frame->initial_ego_pose, frame->frame_id);
    }

    std::vector<VINSFrame*> & getFrames(int drone_id) {
        return drone_frames.at(drone_id);
    }
};
}