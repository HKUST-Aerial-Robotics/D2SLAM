#pragma once
#include <d2common/d2state.hpp>
#include <swarm_msgs/drone_trajectory.hpp>

using namespace D2Common;

namespace D2PGO {
class PGOState : public D2State {
protected:
    std::map<int, std::vector<D2BaseFrame*>> drone_frames;
    std::map<int, Swarm::DroneTrajectory> ego_drone_trajs;
    std::map<int, Eigen::Quaterniond> initial_attitude;

public:
    PGOState(int _self_id, bool _is_4dof = false) :
        D2State(_self_id, _is_4dof) {
        if (is_4dof) {
            printf("[D2PGO] PGOState: is 4dof\n");
        } else {
            printf("[D2PGO] PGOState: is 6dof\n");
        }
        drone_frames[self_id] = std::vector<D2BaseFrame*>();
    }

    void addFrame(const D2BaseFrame & _frame) {
        const Guard lock(state_lock);
        // printf("[D2PGO@%d] PGOState: add frame %ld for drone %d: %s\n", self_id, 
        //         _frame.frame_id, _frame.drone_id, _frame.odom.pose().toStr().c_str());
        all_drones.insert(_frame.drone_id);
        auto * frame = new D2BaseFrame;
        *frame = _frame;
        frame_db[frame->frame_id] = frame;
        if (is_4dof) {
            _frame_pose_state[frame->frame_id] = makeSharedStateArray(POSE4D_SIZE);
            _frame.odom.pose().to_vector_xyzyaw(_frame_pose_state[frame->frame_id]);
        } else {
            _frame_pose_state[frame->frame_id] = makeSharedStateArray(POSE_SIZE);
            _frame_rot_state[frame->frame_id] = makeSharedStateArray(ROTMAT_SIZE);
            _frame_pose_pertub_state[frame->frame_id] = makeSharedStateArray(POSE_EFF_SIZE);
            _frame.odom.pose().to_vector(_frame_pose_state[frame->frame_id]);
            Map<Matrix<state_type, 3, 3, RowMajor>> rot(CheckGetPtr(_frame_rot_state[frame->frame_id]));
            rot = _frame.odom.pose().R();

            Map<Eigen::Vector6d> pose_pertub(CheckGetPtr(_frame_pose_pertub_state[frame->frame_id]));
            pose_pertub.setZero();
            pose_pertub.segment<3>(0) = _frame.T();

            initial_attitude[frame->frame_id] = _frame.odom.att();
        }
        if (drone_frames.find(_frame.drone_id) == drone_frames.end()) {
            drone_frames[_frame.drone_id] = std::vector<D2BaseFrame*>();
            ego_drone_trajs[_frame.drone_id] = Swarm::DroneTrajectory();
        }
        drone_frames.at(_frame.drone_id).push_back(frame);
        ego_drone_trajs[_frame.drone_id].push(frame->stamp, frame->initial_ego_pose, frame->frame_id);
    }

    int size(int drone_id) {
        if (drone_frames.find(drone_id) == drone_frames.end()) {
            return 0;
        }
        return drone_frames.at(drone_id).size();
    }

    std::vector<D2BaseFrame*> & getFrames(int drone_id) {
        return drone_frames.at(drone_id);
    }

    Swarm::DroneTrajectory & getEgomotionTraj(int drone_id) {
        return ego_drone_trajs.at(drone_id);
    }

    const Swarm::DroneTrajectory & getEgomotionTraj(int drone_id) const {
        return ego_drone_trajs.at(drone_id);
    }
    
    bool hasEgomotionTraj(int drone_id) const {
        return ego_drone_trajs.find(drone_id) != ego_drone_trajs.end();
    }

    FrameIdType headId(int drone_id) {
        if (drone_frames.find(drone_id) == drone_frames.end() || 
            drone_frames.at(drone_id).size() == 0) {
            return -1;
        }
        return drone_frames.at(drone_id)[0]->frame_id;
    }

    void syncFromState() {
        const Guard lock(state_lock);
        for (auto it : _frame_pose_state) {
            auto frame_id = it.first;
            if (frame_db.find(frame_id) == frame_db.end()) {
                printf("[D2VINS::D2EstimatorState] Cannot find frame %ld\033[0m\n", frame_id);
            }
            auto frame = frame_db.at(frame_id);
            if (is_4dof) {
                frame->odom.pose().from_vector(it.second, true);
            } else {
                frame->odom.pose().from_vector(it.second);
            }
        }
    }

    void setAttitudeInit(FrameIdType frame_id, const Eigen::Quaterniond & _attitude) {
        initial_attitude[frame_id] = _attitude;
    }

    Eigen::Quaterniond getAttitudeInit(FrameIdType frame_id) {
        return initial_attitude.at(frame_id);
    }

    std::vector<int> getAllDroneIds() {
        std::vector<int> drone_ids;
        for (auto it : drone_frames) {
            drone_ids.push_back(it.first);
        }
        return drone_ids;
    }
};
}