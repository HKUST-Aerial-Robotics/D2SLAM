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

struct D2PGOConfig {
    int self_id = -1;
    PGO_MODE mode = PGO_MODE_NON_DIST;
    D2Common::ARockSolverConfig arock_config;
    ceres::Solver::Options ceres_options;
};

class D2PGO {
protected:
    D2PGOConfig config;
    int self_id;
    PGOState state;
    mutable std::recursive_mutex state_lock;
    std::vector<Swarm::LoopEdge> loops;
    void setupLoopFactors(SolverWrapper * solver);
    void setupEgoMotionFactors(SolverWrapper * solver);
public:
    D2PGO(D2PGOConfig _config):
        config(_config), self_id(_config.self_id), state(_config.self_id) {
    }
    void addFrame(const VINSFrame & frame_desc);
    void addLoop(const Swarm::LoopEdge & loop_info);
    void setStateProperties(ceres::Problem & problem);
    void solve();
};
}