#include <d2vins/utils.hpp>
#include "d2estimator.hpp" 
#include "unistd.h"

namespace D2VINS {
void D2Estimator::inputImu(IMUData data) {
    imubuf.add(data);
    if (!initFirstPoseFlag) {
        return;
    }

    //Propagation current with last Bias.
    
}

bool D2Estimator::tryinitFirstPose(const D2Frontend::VisualImageDescArray & frame) {
    if (imubuf.size() < config.init_imu_num) {
        return false;
    }
    auto q0 = Utility::g2R(imubuf.mean_acc());
    last_odom = Swarm::Odometry(frame.stamp, Swarm::Pose(q0, Vector3d::Zero()));

    VINSFrame first_frame(frame);
    first_frame.odom = last_odom;

    //Easily use the average value as gyrobias now
    first_frame.Bg = imubuf.mean_gyro();
    //Also the ba with average acc - g
    first_frame.Ba = imubuf.mean_acc() - Gravity;
    state.addFrame(first_frame, true);
    
    printf("[D2VINS::D2Estimator] Init pose with IMU: %s\n", last_odom.toStr().c_str());
    printf("[D2VINS::D2Estimator] Gyro bias: %.3f %.3f %.3f\n", first_frame.Bg.x(), first_frame.Bg.y(), first_frame.Bg.z());
    printf("[D2VINS::D2Estimator] Acc  bias: %.3f %.3f %.3f\n\n", first_frame.Ba.x(), first_frame.Ba.y(), first_frame.Ba.z());
    return true;
}

VINSFrame D2Estimator::initFrame(const D2Frontend::VisualImageDescArray & _frame) {
    //First we init corresponding pose for with IMU
    VINSFrame frame(_frame);
    if (config.init_method == D2VINSConfig::INIT_POSE_IMU) {
        auto _imu = imubuf.back(_frame.stamp + state.td);
        frame.odom = _imu.propagation(state.lastFrame());
    } else {
    }

    frame.Ba = state.lastFrame().Ba;
    frame.Bg = state.lastFrame().Bg;
    
    bool is_keyframe = _frame.is_keyframe; //Is keyframe is done in frontend
    state.addFrame(frame, is_keyframe);

    if (config.verbose) {
        printf("[D2VINS::D2Estimator] Initialize VINSFrame with %d: %s\n", 
            config.init_method, frame.toStr().c_str());
    }
    return frame;
}

void D2Estimator::inputImage(D2Frontend::VisualImageDescArray & _frame) {
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

}

Swarm::Odometry D2Estimator::getImuPropagation() const {
    return last_prop_odom;
}

Swarm::Odometry D2Estimator::getOdometry() const {
    return last_odom;
}


}