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
class D2PGO {
protected:
    D2PGOConfig config;
    int self_id;
    int main_id;
    PGOState state;
    mutable std::recursive_mutex state_lock;
    std::vector<Swarm::LoopEdge> all_loops;
    std::set<FrameIdType> used_frames;
    int used_loops_count;
    int solve_count = 0;
    bool updated = false;
    SolverWrapper * solver = nullptr;
    std::vector<Swarm::LoopEdge> used_loops;
    std::map<int, Swarm::DroneTrajectory> ego_motion_trajs;
    SwarmLocalOutlierRejection rejection;

    void saveG2O();
    void setupLoopFactors(SolverWrapper * solver, const std::vector<Swarm::LoopEdge> & good_loops);
    void setupEgoMotionFactors(SolverWrapper * solver);
    void setupEgoMotionFactors(SolverWrapper * solver, int drone_id);
public:
    std::function<void(void)> postsolve_callback;
    std::function<void(const DPGOData & )> bd_data_callback;
    D2PGO(D2PGOConfig _config):
        config(_config), self_id(_config.self_id), main_id(_config.main_id),
        state(_config.self_id, _config.pgo_pose_dof == PGO_POSE_4D),
        rejection(_config.self_id, _config.pcm_rej, ego_motion_trajs) {
    }
    void evalLoop(const Swarm::LoopEdge & loop);
    void addFrame(D2BaseFrame frame_desc);
    void addLoop(const Swarm::LoopEdge & loop_info, bool add_state_by_loop=false);
    void setStateProperties(ceres::Problem & problem);
    bool solve(bool force_solve=false);
    void broadcastData(const DPGOData & data);
    void inputDPGOData(const DPGOData & data);
    void rotInitial(const std::vector<Swarm::LoopEdge> & good_loops);
    std::map<int, Swarm::DroneTrajectory> getOptimizedTrajs();
    std::vector<D2BaseFrame*> getAllLocalFrames();
};
}