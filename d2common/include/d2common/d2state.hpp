#pragma once
#include <set>
#include <map>
#include <d2common/d2vinsframe.h>

namespace D2Common {
class D2State {
protected:
    int self_id;
    std::set<int> all_drones;
    int reference_frame_id = -1;
    std::map<FrameIdType, D2BaseFrame*> frame_db;
    std::map<FrameIdType, state_type*> _frame_pose_state;
    mutable std::recursive_mutex state_lock;
    bool is_4dof = false;
public:
    D2State(int _self_id, bool _is_4dof = false) :
        self_id(_self_id), reference_frame_id(_self_id), is_4dof(_is_4dof) {
    }

    std::set<int> availableDrones() const {
        return all_drones;
    }

    bool hasDrone(int drone_id) const{
        return all_drones.find(drone_id) != all_drones.end();
    }

    bool hasFrame(FrameIdType frame_id) const {
        return frame_db.find(frame_id) != frame_db.end();
    }

    const D2BaseFrame * getFramebyId(int frame_id) const {
        if (frame_db.find(frame_id) == frame_db.end()) {
            printf("\033[0;31m[D2EstimatorState::getFramebyId] Frame %d not found in database\033[0m\n", frame_id);
            assert(true && "Frame not found in database");
        }
        return frame_db.at(frame_id);
    }

    D2BaseFrame * getFramebyId(int frame_id) {
        if (frame_db.find(frame_id) == frame_db.end()) {
            printf("\033[0;31m[D2EstimatorState::getFramebyId] Frame %d not found in database\033[0m\n", frame_id);
            assert(true && "Frame not found in database");
        }
        return frame_db.at(frame_id);
    }

    int getReferenceFrameId() const {
        return reference_frame_id;
    }
    
    void setReferenceFrameId(int _reference_frame_id) {
        reference_frame_id = _reference_frame_id;
    }

    virtual void moveAllPoses(int new_ref_frame_id, const Swarm::Pose & delta_pose) {
        reference_frame_id = new_ref_frame_id;
        const Guard lock(state_lock);
        for (auto it: frame_db) {
            auto frame_id = it.first;
            auto & frame = it.second;
            frame->moveByPose(new_ref_frame_id, delta_pose);
            frame->odom.pose().to_vector(_frame_pose_state.at(frame_id));
        }
    }

    double * getPoseState(FrameIdType frame_id) const {
        const Guard lock(state_lock);
        if (_frame_pose_state.find(frame_id) == _frame_pose_state.end()) {
            printf("\033[0;31m[D2VINS::D2EstimatorState] frame %ld not found\033[0m\n", frame_id);
            assert(false && "Frame not found");
        }
        return _frame_pose_state.at(frame_id);
    }
};
}