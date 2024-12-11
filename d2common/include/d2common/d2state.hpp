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
    std::map<FrameIdType, StatePtr> _frame_pose_state; 

    //This returns the perturb of the frame. per = [T, v], where v is the rotation vector representation of a small R.
    //v = \theta * unit(v)
    //pose = (T, R0*exp(\theta * K)), where K = skewMatrix(unit(v))
    std::map<FrameIdType, StatePtr> _frame_pose_pertub_state;

    //This returns the R matrix pointer which is the rotation of the frame.
    // Note that this rot state is not esstentially a rotation matrix. 
    // To get real rotation matrix from it, use recoverRotationSVD.
    std::map<FrameIdType, StatePtr> _frame_rot_state; 

    mutable std::recursive_mutex state_lock;
    bool is_4dof = false;
public:
    D2State(int _self_id, bool _is_4dof = false) :
        self_id(_self_id), reference_frame_id(_self_id), is_4dof(_is_4dof) {
    }

    std::set<int> availableDrones() const {
        const Guard lock(state_lock);
        return all_drones;
    }

    bool hasDrone(int drone_id) const{
        const Guard lock(state_lock);
        return all_drones.find(drone_id) != all_drones.end();
    }

    bool hasFrame(FrameIdType frame_id) const {
        const Guard lock(state_lock);
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

    StatePtr getPoseState(FrameIdType frame_id) const {
        const Guard lock(state_lock);
        if (_frame_pose_state.find(frame_id) == _frame_pose_state.end()) {
            printf("\033[0;31m[D2State::getPoseState] frame %ld not found\033[0m\n", frame_id);
            assert(false && "Frame not found");
        }
        return _frame_pose_state.at(frame_id);
    }

    StatePtr getRotState(FrameIdType frame_id) const {
        const Guard lock(state_lock);
        if (_frame_rot_state.find(frame_id) == _frame_rot_state.end()) {
            printf("\033[0;31m[D2State::getRotState] frame %ld not found\033[0m\n", frame_id);
            assert(false && "Frame not found");
        }
        return _frame_rot_state.at(frame_id);
    }

    StatePtr getPerturbState(FrameIdType frame_id) const {
        const Guard lock(state_lock);
        if (_frame_pose_pertub_state.find(frame_id) == _frame_pose_pertub_state.end()) {
            printf("\033[0;31m[D2State::getRotState] frame %ld not found\033[0m\n", frame_id);
            assert(false && "Frame not found");
        }
        return _frame_pose_pertub_state.at(frame_id);
    }

    int getSelfId() const {
        return self_id;
    }

    void lock_state() {
        state_lock.lock();
    }

    void unlock_state() {
        state_lock.unlock();
    }
};
}