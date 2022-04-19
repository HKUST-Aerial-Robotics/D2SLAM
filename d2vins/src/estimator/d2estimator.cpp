#include <d2vins/utils.hpp>
#include "d2estimator.hpp" 
#include "unistd.h"
#include <d2vins/factors/imu_factor.h>
#include <d2vins/factors/projectionTwoFrameOneCamFactor.h>

namespace D2VINS {
void D2Estimator::init() {
    state.init(params->camera_extrinsics, params->td_initial);
    ProjectionTwoFrameOneCamFactor::sqrt_info = params->focal_length / 1.5 * Matrix2d::Identity();
}

void D2Estimator::inputImu(IMUData data) {
    imubuf.add(data);
    if (!initFirstPoseFlag) {
        return;
    }

    //Propagation current with last Bias.
    
}

bool D2Estimator::tryinitFirstPose(const D2FrontEnd::VisualImageDescArray & frame) {
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

VINSFrame D2Estimator::initFrame(const D2FrontEnd::VisualImageDescArray & _frame) {
    //First we init corresponding pose for with IMU
    auto _imu = imubuf.back(_frame.stamp + state.td);
    VINSFrame frame(_frame, _imu, state.lastFrame());
    if (params->init_method == D2VINSConfig::INIT_POSE_IMU) {
        frame.odom = _imu.propagation(state.lastFrame());
    } else {
    }

    bool is_keyframe = _frame.is_keyframe; //Is keyframe is done in frontend
    state.addFrame(_frame, frame, is_keyframe);

    if (params->verbose) {
        printf("[D2VINS::D2Estimator] Initialize VINSFrame with %d: %s\n", 
            params->init_method, frame.toStr().c_str());
    }
    return frame;
}

void D2Estimator::inputImage(D2FrontEnd::VisualImageDescArray & _frame) {
    if(!initFirstPoseFlag) {
        printf("[D2VINS::D2Estimator] tryinitFirstPose imu buf %ld\n", imubuf.size());
        initFirstPoseFlag = tryinitFirstPose(_frame);
        return;
    }

    double t_imu_frame = _frame.stamp + state.td;
    while (!imubuf.avaiable(t_imu_frame)) {
        //Wait for IMU
        usleep(2000);
        printf("[D2VINS::D2Estimator] wait for imu...\n");
    }

    auto frame = initFrame(_frame);
    solve();
}

void D2Estimator::solve() {
    state.pre_solve();
    ceres::Problem problem;
    setupImuFactors(problem);
    setupLandmarkFactors(problem);

    if (!params->estimate_td) {
        problem.SetParameterBlockConstant(&state.td);
    }

    if (!params->estimate_extrinsic) {
        for (int i = 0; i < params->camera_num; i ++) {
            problem.SetParameterBlockConstant(state.getExtrinsicState(i));
        }
    }

    //Current no margarin, fix the first pose
    problem.SetParameterBlockConstant(state.getPoseState(state.baseFrame().frame_id));

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;
    state.syncFromState();
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
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(1.0);
    for (auto & lm : lms) {
        auto lm_id = lm.landmark_id;
        auto mea0 = lm.track[0].measurement();
        for (auto i = 1; i < lm.track.size(); i++) {
            auto mea1 = lm.track[0].measurement();
            auto f_td = new ProjectionTwoFrameOneCamFactor(mea0, mea1, lm.track[0].velocity, lm.track[i].velocity,
                lm.track[0].cur_td, lm.track[i].cur_td);
            problem.AddResidualBlock(f_td, loss_function,
                state.getPoseState(lm.track[0].frame_id), 
                state.getPoseState(lm.track[i].frame_id), 
                state.getExtrinsicState(lm.track[0].camera_id),
                state.getLandmarkState(lm_id), &state.td);
        }
    }
}

Swarm::Odometry D2Estimator::getImuPropagation() const {
    return last_prop_odom;
}

Swarm::Odometry D2Estimator::getOdometry() const {
    return last_odom;
}


}