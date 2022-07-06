#include "d2pgo.h"
#include "ARockPGO.hpp"
#include <d2common/solver/RelPoseFactor.hpp>

namespace D2PGO {

void D2PGO::addFrame(const VINSFrame & frame_desc) {
    const Guard lock(state_lock);
    state.addFrame(frame_desc);
}

void D2PGO::addLoop(const Swarm::LoopEdge & loop_info) {
    const Guard lock(state_lock);
    loops.emplace_back(loop_info);
}

void D2PGO::solve() {
    const Guard lock(state_lock);
    SolverWrapper * solver;
    if (config.mode == PGO_MODE_NON_DIST) {
        solver = new CeresSolver(&state, config.ceres_options);
    } else if (config.mode == PGO_MODE_DISTRIBUTED_AROCK) {
        solver = new ARockPGO(&state, config.arock_config);
    }

    setupLoopFactors(solver);
    setupEgoMotionFactors(solver);

    if (config.mode == PGO_MODE_NON_DIST) {
        setStateProperties(solver->getProblem());
    }
}

void D2PGO::setupLoopFactors(SolverWrapper * solver) {
    auto loss_function = new ceres::HuberLoss(1.0);    
    for (auto loop : loops) {
        auto loop_factor = new RelPoseFactor(loop.relative_pose, loop.sqrt_information_matrix());
        auto res_info = RelPoseResInfo::create(loop_factor, 
            loss_function, loop.keyframe_id_a, loop.keyframe_id_b);
    }
}

void D2PGO::setupEgoMotionFactors(SolverWrapper * solver) {
    if (config.mode >= PGO_MODE_DISTRIBUTED_AROCK) {
        //Only set up the ego motion factors for self if we are using the distributed ARock PGO        
        for (auto frame : state.getFrames(self_id)) {
            // auto ego_motion_factor = new RelPoseFactor();
            // auto res_info = RelPoseResInfo::create(ego_motion_factor, nullptr, );
        }
    } else {
        
    }
}

void D2PGO::setStateProperties(ceres::Problem & problem) {

}

}