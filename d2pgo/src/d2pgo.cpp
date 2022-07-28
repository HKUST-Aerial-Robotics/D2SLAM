#include "d2pgo.h"
#include "ARockPGO.hpp"
#include <d2common/solver/RelPoseFactor.hpp>
#include <d2common/solver/pose_local_parameterization.h>
#include <d2common/solver/angle_manifold.h>
#include "../test/posegraph_g2o.hpp"


namespace D2PGO {

void D2PGO::addFrame(const D2BaseFrame & frame_desc) {
    const Guard lock(state_lock);
    state.addFrame(frame_desc);
    printf("[D2PGO@%d]add frame %ld ref %ld pose %s from drone %d\n", self_id, frame_desc.frame_id, frame_desc.reference_frame_id,
        Swarm::Pose(state.getPoseState(frame_desc.frame_id)).toStr().c_str(), frame_desc.drone_id);
    updated = true;
}

void D2PGO::addLoop(const Swarm::LoopEdge & loop_info, bool add_state_by_loop) {
    const Guard lock(state_lock);
    if (loop_info.relative_pose.pos().norm() > config.loop_distance_threshold) {
        printf("[D2PGO@%d]loop distance %.1fm too large, ignore\n", self_id, loop_info.relative_pose.pos().norm());
        return;
    }
    loops.emplace_back(loop_info);
    // printf("Adding edge between keyframe %ld<->%ld drone %d<->%d hasKF %d %d\n ", loop_info.keyframe_id_a, loop_info.keyframe_id_b,
    //         loop_info.id_a, loop_info.id_b, state.hasFrame(loop_info.keyframe_id_a), state.hasFrame(loop_info.keyframe_id_b));
    if (loop_info.relative_pose.pos().norm() > 1.0) {
        ROS_WARN("[D2PGO@%d]loop edge is too far: %s\n", self_id, loop_info.relative_pose.toStr().c_str());
    }
    if (add_state_by_loop) {
        // This is for debug...
        if (state.hasFrame(loop_info.keyframe_id_a) && !state.hasFrame(loop_info.keyframe_id_b)) {
            // Add frame idb to state
            D2BaseFrame frame_desc;
            frame_desc.drone_id = loop_info.id_b;
            frame_desc.frame_id = loop_info.keyframe_id_b;
            frame_desc.reference_frame_id = state.getFramebyId(loop_info.keyframe_id_a)->reference_frame_id;
            // Initialize pose with known pose a and this loop edge
            frame_desc.odom.pose() = state.getFramebyId(loop_info.keyframe_id_a)->odom.pose() * loop_info.relative_pose;
            addFrame(frame_desc);
            // printf("[D2PGO@%d]add frame %ld pose %s from drone %d\n", self_id, frame_desc.frame_id,
            //     frame_desc.odom.pose().toStr().c_str(), frame_desc.drone_id);
        } else if (!state.hasFrame(loop_info.keyframe_id_a) && state.hasFrame(loop_info.keyframe_id_b)) {
            // Add frame idb to state
            D2BaseFrame frame_desc;
            frame_desc.drone_id = loop_info.id_a;
            frame_desc.frame_id = loop_info.keyframe_id_a;
            frame_desc.reference_frame_id = state.getFramebyId(loop_info.keyframe_id_b)->reference_frame_id;
            // Initialize pose with known pose a and this loop edge
            frame_desc.odom.pose() = state.getFramebyId(loop_info.keyframe_id_b)->odom.pose() * loop_info.relative_pose.inverse();
            addFrame(frame_desc);
            // printf("[D2PGO@%d]add frame %ld pose %s from drone %d\n", self_id, frame_desc.frame_id,
            //     frame_desc.odom.pose().toStr().c_str(), frame_desc.drone_id);
        }
    }
    updated = true;
}

void D2PGO::inputDPGOData(const DPGOData & data) {
    if (solver!=nullptr && config.mode == PGO_MODE_DISTRIBUTED_AROCK) {
        static_cast<ARockPGO*>(solver)->inputDPGOData(data);
    }
}

bool D2PGO::solve() {
    const Guard lock(state_lock);
    if (state.size(self_id) < config.min_solve_size || !updated) {
        // printf("[D2PGO] Not enough frames to solve %d.\n", state.size(self_id));
        return false;
    }
    if (config.mode == PGO_MODE_NON_DIST) {
        solver = new CeresSolver(&state, config.ceres_options);
    } else if (config.mode == PGO_MODE_DISTRIBUTED_AROCK) {
        if (solver==nullptr) {
            solver = new ARockPGO(&state, this, config.arock_config);
        } else {
            static_cast<ARockPGO*>(solver)->resetResiduals();
            // solver = new ARockPGO(&state, this, config.arock_config);
        }
    }

    // used_frames.clear();
    used_loops.clear();
    setupLoopFactors(solver);
    if (config.enable_ego_motion) {
        setupEgoMotionFactors(solver);
    }

    if (config.mode == PGO_MODE_NON_DIST) {
        setStateProperties(solver->getProblem());
    }
    if (config.write_g2o) {
        saveG2O();
    }
    auto report = solver->solve();
    state.syncFromState();
    printf("[D2PGO::solve@%d] solve_count %d mode %d total frames %ld loops %d opti_time %.1fms initial cost %.2e final cost %.2e\n", 
            self_id, solve_count, config.mode, used_frames.size(), used_loops_count, report.total_time*1000, 
            report.initial_cost, report.final_cost);
    solve_count ++;
    updated = false;
    return true;
}

void D2PGO::saveG2O() {
    std::vector<D2BaseFrame*> frames;
    for (auto frame_id: used_frames) {
        frames.emplace_back(state.getFramebyId(frame_id));
    }
    printf("[D2PGO::saveG2O@%d] save %ld frames\n", self_id, frames.size());
    write_result_to_g2o(config.g2o_output_path, frames, used_loops, config.g2o_use_raw_data);
}

void D2PGO::evalLoop(const Swarm::LoopEdge & loop) {
    auto factor = new RelPoseFactor4D(loop.relative_pose, loop.get_sqrt_information_4d());
    auto kf_a = state.getFramebyId(loop.keyframe_id_a);
    auto kf_b = state.getFramebyId(loop.keyframe_id_b);
    auto pose_ptr_a = state.getPoseState(loop.keyframe_id_a);
    auto pose_ptr_b = state.getPoseState(loop.keyframe_id_b);
    VectorXd residuals(4);
    auto pose_a = kf_a->odom.pose();
    auto pose_b = kf_b->odom.pose();
    (*factor)(pose_ptr_a, pose_ptr_b, residuals.data());
    printf("Loop %ld->%ld, RelPose %s\n", loop.keyframe_id_a, loop.keyframe_id_b, loop.relative_pose.toStr().c_str()); 
    printf("RelPose            Est %s\n", Swarm::Pose::DeltaPose(pose_a, pose_b).toStr().c_str());
    std::cout << "sqrt_info\n:" << loop.get_sqrt_information_4d() << std::endl;
    printf("PoseA %s PoseB %s residual:", kf_a->odom.pose().toStr().c_str(), kf_b->odom.pose().toStr().c_str());
    std::cout << residuals.transpose() << "\n" << std::endl;
}

void D2PGO::setupLoopFactors(SolverWrapper * solver) {
    used_loops_count = 0;
    auto loss_function = new ceres::HuberLoss(1.0);    
    for (auto loop : loops) {
        ceres::CostFunction * loop_factor = nullptr;
        if (config.pgo_pose_dof == PGO_POSE_4D) {
            loop_factor = RelPoseFactor4D::Create(&loop);
            // this->evalLoop(loop);
        } else {
            loop_factor = RelPoseFactor::Create(&loop);
        }
        if (state.hasFrame(loop.keyframe_id_a) && state.hasFrame(loop.keyframe_id_b)) {
            auto res_info = RelPoseResInfo::create(loop_factor, 
                loss_function, loop.keyframe_id_a, loop.keyframe_id_b, config.pgo_pose_dof == PGO_POSE_4D);
            solver->addResidual(res_info);
            used_frames.insert(loop.keyframe_id_a);
            used_frames.insert(loop.keyframe_id_b);
            used_loops_count ++;
            if (config.write_g2o) {
                used_loops.emplace_back(loop);
            }
            // if (loop.id_a != loop.id_b) 
            //     printf("[D2PGO::setupLoopFactors@%d] add loop %ld->%ld pose: %s\n", 
            //         self_id, loop.keyframe_id_a, loop.keyframe_id_b, loop.relative_pose.toStr().c_str());
        }
    }
}

void D2PGO::setupEgoMotionFactors(SolverWrapper * solver, int drone_id) {
    auto frames = state.getFrames(drone_id);
    auto traj = state.getTraj(drone_id);
    for (int i = 0; i < frames.size() - 1; i ++ ) {
        auto frame_a = frames[i];
        auto frame_b = frames[i + 1];
        Swarm::Pose rel_pose;
        if (config.pgo_pose_dof == PGO_POSE_4D) {
            rel_pose = Swarm::Pose::DeltaPose(frame_a->initial_ego_pose, frame_b->initial_ego_pose, true);
        } else {
            rel_pose = Swarm::Pose::DeltaPose(frame_a->initial_ego_pose, frame_b->initial_ego_pose);
        }
        double len = rel_pose.pos().norm();
        if (len < config.min_cov_len) {
            len = config.min_cov_len;
        }
        Eigen::Matrix6d cov = Eigen::Matrix6d::Zero();
        cov.block<3, 3>(0, 0) = Matrix3d::Identity()*config.pos_covariance_per_meter*len 
            + 0.5*Matrix3d::Identity()*config.yaw_covariance_per_meter*len*len;
        cov.block<3, 3>(3, 3) = Matrix3d::Identity()*config.yaw_covariance_per_meter*len;
        Matrix6d sqrt_info = cov.inverse().cwiseAbs().cwiseSqrt();
        Swarm::LoopEdge loop(frame_a->frame_id, frame_b->frame_id, rel_pose, sqrt_info);
        if (config.pgo_pose_dof == PGO_POSE_4D) {
            auto factor = RelPoseFactor4D::Create(&loop);
            auto res_info = RelPoseResInfo::create(factor, nullptr, frame_a->frame_id, frame_b->frame_id, true);
            solver->addResidual(res_info);
        } else if (config.pgo_pose_dof == PGO_POSE_6D) {
            auto factor = RelPoseFactor::Create(&loop);
            auto res_info = RelPoseResInfo::create(factor, nullptr, frame_a->frame_id, frame_b->frame_id, false);
            solver->addResidual(res_info);
        }
        used_frames.insert(frame_a->frame_id);
        used_frames.insert(frame_b->frame_id);
        if (config.write_g2o) {
            used_loops.emplace_back(loop);
        }
    }
}

void D2PGO::setupEgoMotionFactors(SolverWrapper * solver) {
    if (config.mode == PGO_MODE_NON_DIST) {
        for (auto drone_id : state.availableDrones()) {
            setupEgoMotionFactors(solver, drone_id);
        }
    } else if (config.mode >= PGO_MODE_DISTRIBUTED_AROCK) {
        setupEgoMotionFactors(solver, self_id);
    }
}

void D2PGO::setStateProperties(ceres::Problem & problem) {
    ceres::Manifold* manifold;
    ceres::LocalParameterization * local_parameterization;
    if (config.pgo_pose_dof == PGO_POSE_4D) {
        manifold = PosAngleManifold::Create();
    } else {
        // ceres::EigenQuaternionManifold quat_manifold;
        // ceres::EuclideanManifold<3> euc_manifold;
        // manifold = new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::EigenQuaternionManifold>(euc_manifold, quat_manifold);
        local_parameterization = new PoseLocalParameterization;
    }

    for (auto frame_id : used_frames) {
        auto pointer = state.getPoseState(frame_id);
        if (!problem.HasParameterBlock(pointer)) {
            continue;
        }
        if (config.pgo_pose_dof == PGO_POSE_4D) {
            problem.SetManifold(pointer, manifold);
        } else {
            problem.SetParameterization(pointer, local_parameterization);
        }
    }
    if (config.mode == PGO_MODE_NON_DIST || 
            config.mode >= PGO_MODE_DISTRIBUTED_AROCK && self_id == main_id) {
        auto frame_id = state.headId(self_id);
        auto pointer = state.getPoseState(frame_id);
        problem.SetParameterBlockConstant(pointer);
    } else {
        if (config.mode >= PGO_MODE_DISTRIBUTED_AROCK && self_id != main_id) {
            //In this case we found the first frame of the self drone which reference_frame_id is main_id
            auto frames = state.getFrames(self_id);
            for (auto frame : frames) {
                if (frame->reference_frame_id == main_id) {
                    printf("[D2PGO%d] Set frame %ld constant reference_frame %d \n", self_id, frame->frame_id, frame->reference_frame_id);
                    auto pointer = state.getPoseState(frame->frame_id);
                    problem.SetParameterBlockConstant(pointer);
                    break;
                }
            }
        }
    }
}

std::map<int, Swarm::DroneTrajectory> D2PGO::getOptimizedTrajs() {
    const Guard lock(state_lock);
    std::map<int, Swarm::DroneTrajectory> trajs;
    for (auto drone_id : state.availableDrones()) {
        trajs[drone_id] = Swarm::DroneTrajectory(drone_id, false);
        for (auto frame : state.getFrames(drone_id)) {
            if (used_frames.find(frame->frame_id) == used_frames.end()) {
                continue;
            }
            auto pose = frame->odom.pose();
            if (config.pgo_pose_dof == PGO_POSE_4D) {
                //Then we need to combine the roll pitch from ego motion
                Swarm::Pose ego_pose = frame->initial_ego_pose;
                auto delta_att = ego_pose.att_yaw_only().inverse() * ego_pose.att();
                pose.att() = pose.att()*delta_att;
            }
            trajs[drone_id].push(frame->stamp, pose, frame->frame_id);
        }
    }
    return trajs;
}

void D2PGO::broadcastData(const DPGOData & data) {
    bd_data_callback(data);
}

std::vector<D2BaseFrame*> D2PGO::getAllLocalFrames() {
    const Guard lock(state_lock);
    return state.getFrames(self_id);
}

}