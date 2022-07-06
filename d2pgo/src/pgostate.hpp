#include <d2common/d2state.hpp>
#include <swarm_msgs/drone_trajectory.hpp>

using namespace D2Common;

namespace D2PGO {
class PGOState : public D2State {
protected:
    std::map<int, std::vector<VINSFrame*>> drone_frames;
    std::map<int, Swarm::DroneTrajectory> drone_trajs;

public:
    PGOState(int _self_id, bool _is_4dof = false) :
        D2State(_self_id, _is_4dof) {
        if (is_4dof) {
            printf("[D2PGO] PGOState: is 4dof\n");
        } else {
            printf("[D2PGO] PGOState: is 6dof\n");
        }
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

    Swarm::DroneTrajectory & getTraj(int drone_id) {
        return drone_trajs.at(drone_id);
    }

    FrameIdType headId(int drone_id) {
        if (drone_frames.find(drone_id) == drone_frames.end() || 
            drone_frames.at(drone_id).size() == 0) {
            return -1;
        }
        return drone_frames.at(drone_id)[0]->frame_id;
    }
};
}