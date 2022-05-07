#include <d2vins/d2vins_params.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

namespace D2VINS {
D2VINSConfig * params = nullptr;
Vector3d Gravity;

void initParams(ros::NodeHandle & nh) {
    params = new D2VINSConfig;
    std::string vins_config_path;
    nh.param<std::string>("vins_config_path", vins_config_path, "");
    nh.param<bool>("verbose", params->verbose, false);
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

    //Measurements
    acc_n = fsSettings["acc_n"];
    acc_w = fsSettings["acc_w"];
    gyr_n = fsSettings["gyr_n"];
    gyr_w = fsSettings["gyr_w"];
    depth_sqrt_inf = fsSettings["depth_sqrt_inf"];
    Gravity = Vector3d(0., 0., fsSettings["g_norm"]);

    //Outputs
    fsSettings["output_path"] >> output_folder;
    debug_print_states = (int)fsSettings["debug_print_states"];
    enable_perf_output = (int)fsSettings["enable_perf_output"];
    debug_print_sldwin = (int)fsSettings["debug_print_sldwin"];

    //Solver
    solver_time = fsSettings["max_solver_time"];
    options.max_num_iterations = fsSettings["max_num_iterations"];

    //Estimation
    td_initial = fsSettings["td"];
    estimate_td = (int)fsSettings["estimate_td"];
    estimate_extrinsic = (int)fsSettings["estimate_extrinsic"];
    fuse_dep = (int) fsSettings["fuse_dep"];
    max_depth_to_fuse = fsSettings["max_depth_to_fuse"];
    min_depth_to_fuse = fsSettings["min_depth_to_fuse"];
    always_fixed_first_pose = (int) fsSettings["always_fixed_first_pose"];

    //Multi-drone
    self_id = fsSettings["self_id"];

    //Initialiazation
    init_method = (InitialMethod) (int)fsSettings["init_method"];
    depth_estimate_baseline = fsSettings["depth_estimate_baseline"];

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
    use_llt_for_decompose_A_b = (int)fsSettings["use_llt_for_decompose_A_b"];
    
    for (auto i = 0; i < camera_num; i ++) {
        char name[32] = {0};
        sprintf(name, "body_T_cam%d", i);
        cv::Mat cv_T;
        fsSettings[name] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        camera_extrinsics.push_back(Swarm::Pose(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3)));
    }

    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;// ceres::DENSE_SCHUR;
    options.num_threads = 1;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;// ceres::DOGLEG;
    options.max_solver_time_in_seconds = solver_time;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 1;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    // options.max_solver_time_in_seconds = solver_time;

}

}