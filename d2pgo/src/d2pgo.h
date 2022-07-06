#pragma once
#include "ARockPGO.hpp"
#include <d2common/d2state.hpp>
#include <d2common/d2frontend_types.h>
#include "pgostate.hpp"
#include <swarm_msgs/relative_measurments.hpp>

namespace D2PGO {

enum PGO_MODE {
    PGO_MODE_NON_DIST,
    PGO_MODE_DISTRIBUTED_AROCK
};

enum PGO_POSE_DOF {
    PGO_POSE_4D,
    PGO_POSE_6D
};

struct D2PGOConfig {
    int self_id = -1;
    PGO_MODE mode = PGO_MODE_NON_DIST;
    D2Common::ARockSolverConfig arock_config;
    ceres::Solver::Options ceres_options;
    PGO_POSE_DOF pgo_pose_dof = PGO_POSE_4D;
    double pos_covariance_per_meter = 4e-3;
    double yaw_covariance_per_meter = 4e-5;
    int min_solve_size = 2;
    double min_cov_len = 0.1;
};

class D2PGO {
protected:
    D2PGOConfig config;
    int self_id;
    int main_id;
    PGOState state;
    mutable std::recursive_mutex state_lock;
    std::vector<Swarm::LoopEdge> loops;
    void setupLoopFactors(SolverWrapper * solver);
    void setupEgoMotionFactors(SolverWrapper * solver);
    void setupEgoMotionFactors(SolverWrapper * solver, int drone_id);
    std::set<FrameIdType> used_frames;
    int used_loops_count;
    int solve_count = 0;
    bool updated = false;
public:
    D2PGO(D2PGOConfig _config):
        config(_config), self_id(_config.self_id), main_id(_config.self_id),
        state(_config.self_id, _config.pgo_pose_dof == PGO_POSE_4D) {
    }
    void addFrame(const VINSFrame & frame_desc);
    void addLoop(const Swarm::LoopEdge & loop_info);
    void setStateProperties(ceres::Problem & problem);
    bool solve();
    std::map<int, Swarm::DroneTrajectory> getOptimizedTrajs();
};
}