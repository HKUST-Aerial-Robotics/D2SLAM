#pragma once
#include <d2common/d2basetypes.h>
#include <ceres/problem.h>
#include <d2common/solver/ARock.hpp>

using namespace D2Common;

namespace D2PGO {
//Chordal relaxation algorithm.
struct RotInitConfig {
    bool enable_gravity_prior = true;
    const Vector3d gravity_direction = Vector3d(0, 0, 1); //In FLU frame actually it's sky direction.
    double gravity_sqrt_info = 10;
    bool enable_float32 = true;
    bool enable_pose6d_solver = false;
    int pose6d_iterations = 1;
    int self_id;
};

struct SwarmLocalOutlierRejectionParams {
    bool debug_write_pcm_errors = false;
    bool debug_write_debug = false;
    bool debug_write_pcm_good = false;
    float pcm_thres = 1.635;
    bool enable_pcm = true;
    bool redundant = true;
    bool is_4dof = true;
    bool incremental_pcm = true;
};

struct D2PGOConfig {
    int self_id = -1;
    int main_id = -1;
    PGO_MODE mode = PGO_MODE_NON_DIST;
    D2Common::ARockSolverConfig arock_config;
    ceres::Solver::Options ceres_options;
    PGO_POSE_DOF pgo_pose_dof = PGO_POSE_4D;
    double pos_covariance_per_meter = 4e-3;
    double yaw_covariance_per_meter = 4e-5;
    int min_solve_size = 2;
    double min_cov_len = 0.1;
    bool enable_ego_motion = true;
    double loop_distance_threshold = 1.2;
    bool write_g2o = false;
    std::string g2o_output_path = "";
    bool g2o_use_raw_data = true;
    bool enable_pcm = false;
    bool is_realtime = false;
    bool enable_rotation_initialization = true;
    bool enable_gravity_prior = false;
    bool debug_rot_init_only = false;
    bool pgo_use_autodiff = true;
    bool perturb_mode = true;
    double rot_init_state_eps = 1e-2;
    SwarmLocalOutlierRejectionParams pcm_rej;
    RotInitConfig rot_init_config;
    double rot_init_timeout = 3;
    bool debug_save_g2o_only = false;
};
}