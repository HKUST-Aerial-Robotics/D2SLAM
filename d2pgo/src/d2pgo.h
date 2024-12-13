#pragma once
#include "ARockPGO.hpp"
#include <d2common/d2state.hpp>
#include <d2common/d2frontend_types.h>
#include "pgostate.hpp"
#include <swarm_msgs/relative_measurments.hpp>
#include <d2common/d2pgo_types.h>
#include "../test/posegraph_g2o.hpp"
#include "swarm_outlier_rejection/swarm_outlier_rejection.hpp"
#include "d2pgo_config.h"

namespace D2PGO {
class RotInit;

class D2PGO {
protected:
    D2PGOConfig config;
    int self_id;
    int main_id = 0;
    PGOState state;
    mutable std::recursive_mutex state_lock;
    std::vector<Swarm::LoopEdge> all_loops;
    std::set<FrameIdType> used_frames;
    std::map<int, FrameIdType> used_latest_frames;
    std::map<int, FrameIdType> used_latest_ts;
    int used_loops_count;
    int solve_count = 0;
    bool updated = false;
    bool is_rot_init_convergence = false;
    SolverWrapper * solver = nullptr;
    std::vector<Swarm::LoopEdge> used_loops;
    std::map<int, Swarm::DroneTrajectory> ego_motion_trajs;
    SwarmLocalOutlierRejection rejection;
    RotInit * rot_init = nullptr;
    RotInit * pose6d_init = nullptr;
    std::set<int> available_robots;
    std::set<int> rot_init_finished_robots;
    bool rot_init_finished = false;
    int save_count = 0;

    void saveG2O(bool only_self=false);
    void setupLoopFactors(SolverWrapper * solver, const std::vector<Swarm::LoopEdge> & good_loops);
    void setupEgoMotionFactors(SolverWrapper * solver);
    void setupEgoMotionFactors(SolverWrapper * solver, int drone_id);
    void setupGravityPriorFactors(SolverWrapper * solver);
    bool isMain() const;
    bool isRotInitConvergence() const;
    void waitForRotInitFinish();
public:
    void postPerturbSolve();
    std::function<void(void)> postsolve_callback;
    std::function<void(const DPGOData & )> bd_data_callback;
    std::function<void(const std::string & )> bd_signal_callback;
    D2PGO(D2PGOConfig _config):
        config(_config), self_id(_config.self_id), main_id(_config.main_id),
        state(_config.self_id, _config.pgo_pose_dof == PGO_POSE_4D),
        rejection(_config.self_id, _config.pcm_rej, ego_motion_trajs),
        available_robots{_config.self_id} {
    }
    void evalLoop(const Swarm::LoopEdge & loop);
    void addFrame(const D2BaseFramePtr& frame_desc);
    void addLoop(const Swarm::LoopEdge & loop_info, bool add_state_by_loop=false);
    void setStateProperties(ceres::Problem & problem);
    bool solve_multi(bool force_solve=false);
    bool solve_single();
    void broadcastData(const DPGOData & data);
    void inputDPGOData(const DPGOData & data);
    void inputDPGOsignal(int drone, const std::string & signal);
    void sendSignal(const std::string & signal);
    void rotInitial(const std::vector<Swarm::LoopEdge> & good_loops);
    std::map<int, Swarm::DroneTrajectory> getOptimizedTrajs();
    std::vector<D2BaseFramePtr> getAllLocalFrames();
    void setAvailableRobots(const std::set<int> & _available_robots) {
        available_robots = _available_robots;
    }
    int getReferenceFrameId() const {
        return state.getReferenceFrameId();
    }
    std::map<int, Swarm::Odometry> getPredictedOdoms() const;
};
}