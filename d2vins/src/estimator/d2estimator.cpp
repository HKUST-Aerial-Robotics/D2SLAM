#include <d2vins/utils.hpp>
#include "d2estimator.hpp" 
#include "unistd.h"
#include "../factors/imu_factor.h"
#include "../factors/depth_factor.h"
#include "../factors/prior_factor.h"
#include "../factors/projectionTwoFrameOneCamDepthFactor.h"
#include "../factors/projectionTwoFrameOneCamFactor.h"
#include "../factors/projectionOneFrameTwoCamFactor.h"
#include "../factors/projectionTwoFrameTwoCamFactor.h"
#include "../factors/projectionTwoFrameOneCamFactorNoTD.h"
#include "../factors/pose_local_parameterization.h"
#include <d2frontend/utils.h>
#include "marginalization/marginalization.hpp"

namespace D2VINS {
void D2Estimator::init(ros::NodeHandle & nh) {
    state.init(params->camera_extrinsics, params->td_initial);
    ProjectionTwoFrameOneCamFactor::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameOneCamDepthFactor::sqrt_info = params->focal_length / 1.5 * Matrix3d::Identity();
    ProjectionTwoFrameOneCamDepthFactor::sqrt_info(2,2) = params->depth_sqrt_inf;
    visual.init(nh, this);
    printf("[D2Estimator::init] init done estimator on drone %d\n", params->self_id);
    for (auto cam_id : state.getAvailableCameraIds()) {
        Swarm::Pose ext = state.getExtrinsic(cam_id);
        printf("[D2VINS::D2Estimator] extrinsic %d: %s\n", cam_id, ext.toStr().c_str());
    }
}

void D2Estimator::inputImu(IMUData data) {
    imubuf.add(data);
    if (!initFirstPoseFlag) {
        return;
    }
    //Propagation current with last Bias.
}

bool D2Estimator::tryinitFirstPose(const VisualImageDescArray & frame) {
    if (imubuf.size() < params->init_imu_num) {
        return false;
    }
    auto q0 = Utility::g2R(imubuf.mean_acc());
    last_odom = Swarm::Odometry(frame.stamp, Swarm::Pose(q0, Vector3d::Zero()));

    //Easily use the average value as gyrobias now
    //Also the ba with average acc - g
    VINSFrame first_frame(frame, imubuf.mean_acc() - Gravity, imubuf.mean_gyro());
    first_frame.odom = last_odom;

    state.addFrame(frame, first_frame, true);
    
    printf("\033[0;32m[D2VINS::D2Estimator] Init pose with IMU: %s\n", last_odom.toStr().c_str());
    printf("\033[0;32m[D2VINS::D2Estimator] Gyro bias: %.3f %.3f %.3f\n", first_frame.Bg.x(), first_frame.Bg.y(), first_frame.Bg.z());
    printf("\033[0;32m[D2VINS::D2Estimator] Acc  bias: %.3f %.3f %.3f\033[0m\n\n", first_frame.Ba.x(), first_frame.Ba.y(), first_frame.Ba.z());
    return true;
}

std::pair<bool, Swarm::Pose> D2Estimator::initialFramePnP(const VisualImageDescArray & frame, const Swarm::Pose & pose) {
    //Only use first image for initialization.
    auto & image = frame.images[0];
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;
    for (auto & lm: image.landmarks) {
        auto & lm_id = lm.landmark_id;
        if (state.hasLandmark(lm_id)) {
            auto & est_lm = state.getLandmarkbyId(lm_id);
            if (est_lm.flag >= LandmarkFlag::INITIALIZED) {
                pts3d.push_back(cv::Point3f(est_lm.position.x(), est_lm.position.y(), est_lm.position.z()));
                pts2d.push_back(cv::Point2f(lm.pt2d_norm.x(), lm.pt2d_norm.y()));
            }
        }
    }

    if (pts3d.size() < params->pnp_min_inliers) {
        return std::make_pair(false, Swarm::Pose());
    }

    cv::Mat inliers;
    cv::Mat D, rvec, t;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    D2FrontEnd::PnPInitialFromCamPose(pose*state.getExtrinsic(0), rvec, t);
    // bool success = cv::solvePnP(pts3d, pts2d, K, D, rvec, t, true);
    bool success = cv::solvePnPRansac(pts3d, pts2d, K, D, rvec, t, true, params->pnp_iteratives,  3, 0.99,  inliers);
    auto pose_cam = D2FrontEnd::PnPRestoCamPose(rvec, t);
    auto pose_imu = pose_cam*state.getExtrinsic(0).inverse();
    // printf("[D2VINS::D2Estimator] PnP initial %s final %s points %d\n", pose.toStr().c_str(), pose_imu.toStr().c_str(), pts3d.size());
    return std::make_pair(success, pose_imu);
}

void D2Estimator::addFrame(const VisualImageDescArray & _frame) {
    //First we init corresponding pose for with IMU
    state.clearFrame();
    if (state.size() > 0) {
        imubuf.pop(state.firstFrame().stamp + state.td);
    }
    auto _imu = imubuf.periodIMU(state.lastFrame().stamp + state.td, _frame.stamp + state.td);
    assert(_imu.size() > 0 && "IMU buffer is empty");
    if (fabs(_imu[_imu.size()-1].t - _frame.stamp - state.td) > params->max_imu_time_err && frame_count > 10) {
        printf("\033[0;31m[D2VINS::D2Estimator] Too large time difference %.3f\n", _imu[_imu.size()-1].t - _frame.stamp - state.td);
        printf("\033[0;31m[D2VINS::D2Estimator] Prev frame  %.3f cur   %.3f td %.1fms\n", state.lastFrame().stamp + state.td, _frame.stamp + state.td, state.td*1000);
        printf("\033[0;31m[D2VINS::D2Estimator] Imu t_start %.3f t_end %.3f num %ld t_last %.3f\033[0m\n", _imu[0].t, _imu[_imu.size()-1].t, _imu.size(), imubuf[imubuf.size()-1].t);
    }
    VINSFrame frame(_frame, _imu, state.lastFrame());
    if (params->init_method == D2VINSConfig::INIT_POSE_IMU) {
        frame.odom = _imu.propagation(state.lastFrame());
    } else {
        auto odom_imu = _imu.propagation(state.lastFrame());
        auto pnp_init = initialFramePnP(_frame, state.lastFrame().odom.pose());
        if (!pnp_init.first) {
            //Use IMU
            printf("\033[0;31m[D2VINS::D2Estimator] Initialization failed, use IMU instead.\033[0m\n");
        } else {
            odom_imu.pose() = pnp_init.second;
        }
        frame.odom = odom_imu;
    }

    bool is_keyframe = _frame.is_keyframe; //Is keyframe is done in frontend
    state.addFrame(_frame, frame, is_keyframe);

    if (params->verbose || params->debug_print_states) {
        printf("[D2VINS::D2Estimator] Initialize VINSFrame with %d: %s\n", 
            params->init_method, frame.toStr().c_str());
    }
}

void D2Estimator::inputImage(VisualImageDescArray & _frame) {
    if(!initFirstPoseFlag) {
        printf("[D2VINS::D2Estimator] tryinitFirstPose imu buf %ld\n", imubuf.size());
        initFirstPoseFlag = tryinitFirstPose(_frame);
        return;
    }

    double t_imu_frame = _frame.stamp + state.td;
    while (!imubuf.available(t_imu_frame)) {
        //Wait for IMU
        usleep(2000);
        printf("[D2VINS::D2Estimator] wait for imu...\n");
    }

    addFrame(_frame);
    if (state.size() >= params->min_solve_frames) {
        solve();
    } else {
        //Presolve only for initialization.
        state.preSolve();
    }
    frame_count ++;
}

void D2Estimator::setStateProperties(ceres::Problem & problem) {
    // ceres::EigenQuaternionManifold quat_manifold;
    // ceres::EuclideanManifold<3> euc_manifold;
    // auto pose_manifold = new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::EigenQuaternionManifold>(euc_manifold, quat_manifold);
    auto pose_local_param = new PoseLocalParameterization;
    //set LocalParameterization
    for (size_t i = 0; i < state.size(); i ++ ) {
        auto & frame_a = state.getFrame(i);
        // problem.SetManifold(state.getPoseState(frame_a.frame_id), pose_manifold);
        problem.SetParameterization(state.getPoseState(frame_a.frame_id), pose_local_param);
    }

    for (auto cam_id: used_camera_sets) {
        if (!params->estimate_extrinsic || state.size() < params->max_sld_win_size) {
            problem.SetParameterBlockConstant(state.getExtrinsicState(cam_id));
        } else {
            // problem.SetManifold(state.getExtrinsicState(cam_id), pose_manifold);
            problem.SetParameterization(state.getExtrinsicState(cam_id), pose_local_param);
        }
    }

    if (!params->estimate_td) {
        problem.SetParameterBlockConstant(state.getTdState(0));
    }

    if (!state.getPrior() || params->always_fixed_first_pose) {
        problem.SetParameterBlockConstant(state.getPoseState(state.firstFrame().frame_id));
    }
}

void D2Estimator::solve() {
    marginalizer = new Marginalizer(&state);
    state.setMarginalizer(marginalizer);
    solve_count ++;
    state.preSolve();
    used_camera_sets.clear();
    problem = new ceres::Problem();
    setupImuFactors(*problem);
    setupLandmarkFactors(*problem);
    setupPriorFactor(*problem);
    setStateProperties(*problem);

    ceres::Solver::Summary summary;
    // params->options.?
    ceres::Solve(params->options, problem, &summary);
    state.syncFromState();
    last_odom = state.lastFrame().odom;

    //Now do some statistics
    static double sum_time = 0;
    static double sum_iteration = 0;
    static double sum_cost = 0;
    sum_time += summary.total_time_in_seconds;
    sum_iteration += summary.num_successful_steps + summary.num_unsuccessful_steps;
    sum_cost += summary.final_cost;

    if (params->enable_perf_output) {
        std::cout << summary.BriefReport() << std::endl;
        printf("[D2VINS] average time %.3fms, average iteration %.3f, average cost %.3f\n", 
            sum_time*1000/solve_count, sum_iteration/solve_count, sum_cost/solve_count);
    }

    printf("[D2VINS] solve_count %d landmarks %d odom %s td %.1fms opti_time %.1fms\n", solve_count, 
        current_landmark_num, last_odom.toStr().c_str(), state.td*1000, summary.total_time_in_seconds*1000);

    //Reprogation
    auto _imu = imubuf.back(state.lastFrame().stamp + state.td);
    last_prop_odom = _imu.propagation(state.lastFrame());
    visual.postSolve();

    if (params->debug_print_states || params->debug_print_sldwin) {
        state.printSldWin();
    }

    if (summary.termination_type == ceres::FAILURE)  {
        std::cout << summary.message << std::endl;
        exit(1);
    }

    // if (state.getPrior() != nullptr) {
    //     exit(0);
    // }
}

void D2Estimator::setupImuFactors(ceres::Problem & problem) {
    for (size_t i = 0; i < state.size() - 1; i ++ ) {
        auto & frame_a = state.getFrame(i);
        auto & frame_b = state.getFrame(i + 1);
        auto pre_integrations = frame_b.pre_integrations; //Prev to current
        IMUFactor* imu_factor = new IMUFactor(pre_integrations);
        problem.AddResidualBlock(imu_factor, nullptr, 
            state.getPoseState(frame_a.frame_id), state.getSpdBiasState(frame_a.frame_id), 
            state.getPoseState(frame_b.frame_id), state.getSpdBiasState(frame_b.frame_id));
        if (params->always_fixed_first_pose) {
            //At this time we fix the first pose and ignore the margin of this imu factor to achieve better numerical stability
            continue;
        }
        marginalizer->addImuResidual(imu_factor, frame_a.frame_id, frame_b.frame_id);
    }
}

void D2Estimator::setupLandmarkFactors(ceres::Problem & problem) {
    auto lms = state.availableLandmarkMeasurements();
    current_landmark_num = lms.size();
    auto loss_function = new ceres::HuberLoss(1.0);    
    std::vector<int> keyframe_measurements(state.size(), 0);
    
    for (auto lm : lms) {
        auto lm_id = lm.landmark_id;
        auto firstObs = lm.track[0];
        auto mea0 = firstObs.measurement();
        keyframe_measurements[state.getPoseIndex(firstObs.frame_id)] ++;
        state.getLandmarkbyId(lm_id).solver_flag = LandmarkSolverFlag::SOLVED;
        if (firstObs.depth_mea && params->fuse_dep && 
                firstObs.depth < params->max_depth_to_fuse &&
                firstObs.depth > params->min_depth_to_fuse) {
            auto f_dep = OneFrameDepth::Create(firstObs.depth);
            problem.AddResidualBlock(f_dep, loss_function, state.getLandmarkState(lm_id));
            marginalizer->addDepthResidual(f_dep, loss_function, firstObs.frame_id, lm_id);
        }
        for (auto i = 1; i < lm.track.size(); i++) {
            auto mea1 = lm.track[i].measurement();
            ceres::CostFunction * f_td = nullptr;
            if (lm.track[i].depth_mea && params->fuse_dep && 
                lm.track[i].depth < params->max_depth_to_fuse && 
                lm.track[i].depth > params->min_depth_to_fuse) {
                f_td = new ProjectionTwoFrameOneCamDepthFactor(mea0, mea1, firstObs.velocity, lm.track[i].velocity,
                    firstObs.cur_td, lm.track[i].cur_td, lm.track[i].depth);
            } else {
                f_td = new ProjectionTwoFrameOneCamFactor(mea0, mea1, firstObs.velocity, lm.track[i].velocity,
                    firstObs.cur_td, lm.track[i].cur_td);
            }
            problem.AddResidualBlock(f_td, loss_function,
                state.getPoseState(firstObs.frame_id), 
                state.getPoseState(lm.track[i].frame_id), 
                state.getExtrinsicState(firstObs.camera_id),
                state.getLandmarkState(lm_id), state.getTdState(lm.track[i].camera_id));
            marginalizer->addLandmarkResidual(f_td, loss_function,
                firstObs.frame_id, lm.track[i].frame_id, lm_id, firstObs.camera_id, true);
            keyframe_measurements[state.getPoseIndex(lm.track[i].frame_id)] ++;
            used_camera_sets.insert(firstObs.camera_id);
        }

        for (auto l_fm : lm.track_r) {
            auto mea1 = l_fm.measurement();
            if (l_fm.frame_id == firstObs.frame_id) {
                auto f_td = new ProjectionOneFrameTwoCamFactor(mea0, mea1, firstObs.velocity, 
                    l_fm.velocity, firstObs.cur_td, l_fm.cur_td);
                problem.AddResidualBlock(f_td, nullptr,
                    state.getExtrinsicState(firstObs.camera_id),
                    state.getExtrinsicState(l_fm.camera_id),
                    state.getLandmarkState(lm_id), state.getTdState(l_fm.camera_id));
                marginalizer->addLandmarkResidualOneFrameTwoCam(f_td, loss_function,
                    firstObs.frame_id, lm_id, firstObs.camera_id, l_fm.camera_id);
            } else {
                auto f_td = new ProjectionTwoFrameTwoCamFactor(mea0, mea1, firstObs.velocity, 
                    l_fm.velocity, firstObs.cur_td, l_fm.cur_td);
                problem.AddResidualBlock(f_td, loss_function,
                    state.getPoseState(firstObs.frame_id), 
                    state.getPoseState(l_fm.frame_id), 
                    state.getExtrinsicState(firstObs.camera_id),
                    state.getExtrinsicState(l_fm.camera_id),
                    state.getLandmarkState(lm_id), state.getTdState(l_fm.camera_id));
                marginalizer->addLandmarkResidualTwoFrameTwoCam(f_td, loss_function,
                    firstObs.frame_id, l_fm.frame_id, lm_id, firstObs.camera_id, l_fm.camera_id);
            }
            used_camera_sets.insert(l_fm.camera_id);
        }
        problem.SetParameterLowerBound(state.getLandmarkState(lm_id), 0, params->min_inv_dep);
    }

    //Check the measurements number of each keyframe
    for (auto i = 0; i < state.size(); i++) {
        if (keyframe_measurements[i] < params->min_measurements_per_keyframe) {
            printf("\033[0;31m[D2VINS::D2Estimator] keyframe index %d frame_id %ld has only %d measurements\033[0m\n", 
                i, state.getFrame(i).frame_id, keyframe_measurements[i]);
        }
    }
}

void D2Estimator::setupPriorFactor(ceres::Problem & problem) {
    auto prior_factor = state.getPrior();
    if (prior_factor != nullptr) {
        problem.AddResidualBlock(prior_factor, nullptr, prior_factor->getKeepParamsPointers());
        marginalizer->addPrior(prior_factor);
    }
}

Swarm::Odometry D2Estimator::getImuPropagation() const {
    return last_prop_odom;
}

Swarm::Odometry D2Estimator::getOdometry() const {
    return last_odom;
}

D2EstimatorState & D2Estimator::getState() {
    return state;
}

}