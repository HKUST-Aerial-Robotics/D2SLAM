#include <d2vins/utils.hpp>
#include "d2estimator.hpp" 
#include "unistd.h"
#include "../factors/imu_factor.h"
#include "../factors/depth_factor.h"
#include "../factors/projectionTwoFrameOneCamFactor.h"
#include <d2frontend/utils.h>

namespace D2VINS {
void D2Estimator::init(ros::NodeHandle & nh) {
    state.init(params->camera_extrinsics, params->td_initial);
    ProjectionTwoFrameOneCamFactor::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
    visual.init(nh, this);
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
    
    printf("[D2VINS::D2Estimator] Init pose with IMU: %s\n", last_odom.toStr().c_str());
    printf("[D2VINS::D2Estimator] Gyro bias: %.3f %.3f %.3f\n", first_frame.Bg.x(), first_frame.Bg.y(), first_frame.Bg.z());
    printf("[D2VINS::D2Estimator] Acc  bias: %.3f %.3f %.3f\n\n", first_frame.Ba.x(), first_frame.Ba.y(), first_frame.Ba.z());
    return true;
}

std::pair<bool, Swarm::Pose> D2Estimator::initialFramePnP(const VisualImageDescArray & frame) {
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
    bool success = cv::solvePnPRansac(pts3d, pts2d, K, D, rvec, t, false,  params->pnp_iteratives,  3, 0.99,  inliers);
    auto pose_cam = D2FrontEnd::PnPRestoCamPose(rvec, t);
    auto pose_imu = pose_cam*state.getExtrinsic(0).inverse();
    return std::make_pair(success, pose_imu);
}

void D2Estimator::addFrame(const VisualImageDescArray & _frame) {
    //First we init corresponding pose for with IMU
    state.clearFrame();
    if (state.size() > 0) {
        imubuf.pop(state.firstFrame().stamp + state.td);
    }
    auto _imu = imubuf.periodIMU(state.lastFrame().stamp + state.td, _frame.stamp + state.td);
    VINSFrame frame(_frame, _imu, state.lastFrame());
    if (params->init_method == D2VINSConfig::INIT_POSE_IMU) {
        frame.odom = _imu.propagation(state.lastFrame());
    } else {
        auto odom_imu = _imu.propagation(state.lastFrame());
        auto pnp_init = initialFramePnP(_frame);
        if (!pnp_init.first) {
            //Use IMU
            printf("[D2VINS::D2Estimator] Initialization failed, use IMU instead.\n");
        } else {
            odom_imu.pose() = pnp_init.second;
        }
        frame.odom = odom_imu;

    }

    bool is_keyframe = _frame.is_keyframe; //Is keyframe is done in frontend
    state.addFrame(_frame, frame, is_keyframe);

    // if (params->verbose) {
        printf("[D2VINS::D2Estimator] Initialize VINSFrame with %d: %s\n", 
            params->init_method, frame.toStr().c_str());
    // }
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
    if (state.size() > params->min_solve_frames) {
        solve();
    } else {
        //Presolve only for initialization.
        state.pre_solve();
    }
    frame_count ++;
}

void D2Estimator::setStateProperties(ceres::Problem & problem) {
    if (!params->estimate_td) {
        problem.SetParameterBlockConstant(&state.td);
    }

    ceres::EigenQuaternionManifold quat_manifold;
    ceres::EuclideanManifold<3> euc_manifold;
    auto pose_manifold = new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::EigenQuaternionManifold>(euc_manifold, quat_manifold);
   
    //set LocalParameterization
    for (size_t i = 0; i < state.size(); i ++ ) {
        auto & frame_a = state.getFrame(i);
        problem.SetManifold(state.getPoseState(frame_a.frame_id), pose_manifold);
    }

    for (int i = 0; i < params->camera_num; i ++) {
        if (!params->estimate_extrinsic) {
            problem.SetParameterBlockConstant(state.getExtrinsicState(i));
        } else {
            problem.SetManifold(state.getExtrinsicState(i), pose_manifold);
        }
    }

    //Current no margarin, fix the first pose
    problem.SetParameterBlockConstant(state.getPoseState(state.firstFrame().frame_id));
}

void D2Estimator::solve() {
    solve_count ++;
    state.pre_solve();
    ceres::Problem problem;
    setupImuFactors(problem);
    setupLandmarkFactors(problem);
    setStateProperties(problem);

    ceres::Solver::Summary summary;
    ceres::Solve(params->options, &problem, &summary);
    // std::cout << summary.FullReport() << std::endl;
    // std::cout << summary.BriefReport() << std::endl;
    state.syncFromState();
    last_odom = state.lastFrame().odom;

    printf("[D2VINS] solve_count %d frame_count %d odom %s td %.1fms\n", solve_count, frame_count, last_odom.toStr().c_str(), state.td*1000);

    //Reprogation
    auto _imu = imubuf.back(state.lastFrame().stamp + state.td);
    last_prop_odom = _imu.propagation(state.lastFrame());
    visual.postSolve();

    state.printSldWin();

    // if (solve_count > 3) {
    //     exit(0);
    // }
}

void D2Estimator::setupImuFactors(ceres::Problem & problem) {
    for (size_t i = 0; i < state.size() - 1; i ++ ) {
        auto & frame_a = state.getFrame(i);
        auto & frame_b = state.getFrame(i + 1);
        auto pre_integrations = frame_b.pre_integrations; //Prev to cuurent
        IMUFactor* imu_factor = new IMUFactor(pre_integrations);
        problem.AddResidualBlock(imu_factor, NULL, 
            state.getPoseState(frame_a.frame_id), state.getSpdBiasState(frame_a.frame_id), 
            state.getPoseState(frame_b.frame_id), state.getSpdBiasState(frame_b.frame_id));
    }
}

void D2Estimator::setupLandmarkFactors(ceres::Problem & problem) {
    auto lms = state.availableLandmarkMeasurements();
    auto loss_function = nullptr;//new ceres::HuberLoss(1.0);    
    for (auto & lm : lms) {
        auto lm_id = lm.landmark_id;
        auto & firstObs = lm.track[0];
        auto mea0 = firstObs.measurement();
        if (firstObs.depth_mea) {
            auto f_dep = OneFrameDepth::Create(firstObs.depth);
            problem.AddResidualBlock(f_dep, nullptr, state.getLandmarkState(lm_id));
            // printf("[D2VINS::D2Estimator] add depth factor %d intial %f mea %f\n", 
            //     lm_id, *state.getLandmarkState(lm_id), firstObs.depth);
        }
        for (auto i = 1; i < lm.track.size(); i++) {
            auto mea1 = firstObs.measurement();
            auto f_td = new ProjectionTwoFrameOneCamFactor(mea0, mea1, firstObs.velocity, lm.track[i].velocity,
                firstObs.cur_td, lm.track[i].cur_td);
            problem.AddResidualBlock(f_td, loss_function,
                state.getPoseState(firstObs.frame_id), 
                state.getPoseState(lm.track[i].frame_id), 
                state.getExtrinsicState(firstObs.camera_id),
                state.getLandmarkState(lm_id), &state.td);
        }

        problem.SetParameterLowerBound(state.getLandmarkState(lm_id), 0, params->min_inv_dep);
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