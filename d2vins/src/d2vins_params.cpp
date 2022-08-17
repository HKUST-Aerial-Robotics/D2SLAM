#include "d2vins_params.hpp"
#include <d2common/integration_base.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <d2common/solver/ConsensusSolver.hpp>
#include <d2frontend/d2frontend_params.h>

using namespace D2Common;

namespace D2VINS {
D2VINSConfig * params = nullptr;

void initParams(ros::NodeHandle & nh) {
    params = new D2VINSConfig;
    std::string vins_config_path;
    nh.param<std::string>("vins_config_path", vins_config_path, "");
    nh.param<bool>("verbose", params->verbose, false);
    nh.param<std::string>("lcm_uri", params->lcm_uri, "udpm://224.0.0.251:7667?ttl=1");
    nh.param<int>("self_id", params->self_id, 0);
    params->init(vins_config_path);
}

void D2VINSConfig::init(const std::string & config_file) {
    printf("[D2VINS::D2VINSConfig] readConfig from file %s\n", config_file.c_str());
    cv::FileStorage fsSettings;
    try {
        fsSettings.open(config_file.c_str(), cv::FileStorage::READ);
    } catch(cv::Exception ex) {
        std::cerr << "ERROR:" << ex.what() << " Can't open config file" << std::endl;
        exit(-1);
    }

    //Inputs
    camera_num = fsSettings["num_of_cam"];
    IMU_FREQ = fsSettings["imu_freq"];
    max_imu_time_err = 1.5/IMU_FREQ;
    frame_step = fsSettings["frame_step"];
    imu_topic = (std::string) fsSettings["imu_topic"];

    //Measurements
    acc_n = fsSettings["acc_n"];
    acc_w = fsSettings["acc_w"];
    gyr_n = fsSettings["gyr_n"];
    gyr_w = fsSettings["gyr_w"];
    Eigen::Matrix<double, 18, 18> noise = Eigen::Matrix<double, 18, 18>::Zero();
    noise.block<3, 3>(0, 0) =  (params->acc_n * params->acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(3, 3) =  (params->gyr_n * params->gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(6, 6) =  (params->acc_n * params->acc_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(9, 9) =  (params->gyr_n * params->gyr_n) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(12, 12) =  (params->acc_w * params->acc_w) * Eigen::Matrix3d::Identity();
    noise.block<3, 3>(15, 15) =  (params->gyr_w * params->gyr_w) * Eigen::Matrix3d::Identity();
    IntegrationBase::noise = noise;
    
    depth_sqrt_inf = fsSettings["depth_sqrt_inf"];
    IMUBuffer::Gravity = Vector3d(0., 0., fsSettings["g_norm"]);

    //Outputs
    fsSettings["output_path"] >> output_folder;
    debug_print_states = (int)fsSettings["debug_print_states"];
    enable_perf_output = (int)fsSettings["enable_perf_output"];
    debug_print_sldwin = (int)fsSettings["debug_print_sldwin"];
    debug_write_margin_matrix = (int)fsSettings["debug_write_margin_matrix"];
    
    //Estimation
    td_initial = fsSettings["td"];
    estimate_td = (int)fsSettings["estimate_td"];
    estimate_extrinsic = (int)fsSettings["estimate_extrinsic"];
    fuse_dep = (int) fsSettings["fuse_dep"];
    max_depth_to_fuse = fsSettings["max_depth_to_fuse"];
    min_depth_to_fuse = fsSettings["min_depth_to_fuse"];
    always_fixed_first_pose = (int) fsSettings["always_fixed_first_pose"];

    //Multi-drone
    estimation_mode = (ESTIMATION_MODE) (int) fsSettings["estimation_mode"];

    //Initialiazation
    init_method = (InitialMethod) (int)fsSettings["init_method"];
    depth_estimate_baseline = fsSettings["depth_estimate_baseline"];
    tri_max_err = fsSettings["tri_max_err"];
    
    //Sliding window
    max_sld_win_size = fsSettings["max_sld_win_size"];
    landmark_estimate_tracks = fsSettings["landmark_estimate_tracks"];
    min_solve_frames = fsSettings["min_solve_frames"];

    //Outlier rejection
    perform_outlier_rejection_num = fsSettings["perform_outlier_rejection_num"];
    landmark_outlier_threshold = fsSettings["thres_outlier"];

    //Marginalization
    margin_sparse_solver = (int)fsSettings["margin_sparse_solver"];
    enable_marginalization = (int)fsSettings["enable_marginalization"];
    remove_base_when_margin_remote = (int)fsSettings["remove_base_when_margin_remote"];
    
    camera_extrinsics = D2FrontEnd::params->extrinsics;

    //Solver
    solver_time = fsSettings["max_solver_time"];
    // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;// ceres::DENSE_SCHUR;
    // options.num_threads = 1;
    // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;// ceres::DOGLEG;
    // options.max_solver_time_in_seconds = solver_time;
    ceres_options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres_options.num_threads = 1;
    ceres_options.trust_region_strategy_type = ceres::DOGLEG;
    ceres_options.max_solver_time_in_seconds = solver_time;
    ceres_options.max_num_iterations = fsSettings["max_num_iterations"];

    //Consenus Solver
    consensus_config = new ConsensusSolverConfig;
    consensus_config->ceres_options.linear_solver_type = ceres::DENSE_SCHUR;
    consensus_config->ceres_options.num_threads = 1;
    consensus_config->ceres_options.trust_region_strategy_type = ceres::DOGLEG;
    consensus_config->max_steps = fsSettings["consensus_max_steps"];
    consensus_config->ceres_options.max_num_iterations = ceres_options.max_num_iterations/consensus_config->max_steps;
    consensus_config->ceres_options.max_solver_time_in_seconds = solver_time/consensus_config->max_steps;
    consensus_config->self_id = self_id;
    consensus_config->timout_wait_sync = fsSettings["timout_wait_sync"];
    consensus_config->rho_landmark = fsSettings["rho_landmark"];
    consensus_config->rho_frame_T = fsSettings["rho_frame_T"];
    consensus_config->rho_frame_theta = fsSettings["rho_frame_theta"];
    consensus_config->relaxation_alpha = fsSettings["relaxation_alpha"];
    consensus_config->sync_with_main = (int) fsSettings["consensus_sync_with_main"];
    consensus_sync_to_start = (int) fsSettings["consensus_sync_to_start"];

}

}