#include <d2vins/utils.hpp>
#include "d2estimator.hpp" 
#include "unistd.h"

namespace D2VINS {
void D2Estimator::inputImu(IMUData data) {
    imubuf.add(data);
    if (!initFirstPoseFlag) {
        return;
    }
}

bool D2Estimator::tryinitFirstPose(const D2Frontend::VisualImageDescArray & frame) {
    if (imubuf.size() < config.init_imu_num) {
        return false;
    }
    auto q0 = g2R(imubuf.mean_acc());
    last_pose = Swarm::Pose(q0, Vector3d::Zero());

    VINSFrame firstFrame(frame);
    firstFrame.pose = last_pose;

    //Easily use the average value as gyrobias now
    firstFrame.Bg = imubuf.mean_gyro();
    //Also the ba with average acc - g
    firstFrame.Ba = imubuf.mean_acc() - config.Gravity;
    
    printf("[D2VINS::D2Estimator] Init pose with IMU: %s\n", last_pose.tostr().c_str());
    printf("[D2VINS::D2Estimator] Gyro bias: %.3f %.3f %.3f\n", firstFrame.Bg.x(), firstFrame.Bg.y(), firstFrame.Bg.z());
    printf("[D2VINS::D2Estimator] Acc  bias: %.3f %.3f %.3f\n\n", firstFrame.Ba.x(), firstFrame.Ba.y(), firstFrame.Ba.z());
    return true;
}

void D2Estimator::inputImage(D2Frontend::VisualImageDescArray & frame) {
    if(!initFirstPoseFlag) {
        printf("[D2VINS::D2Estimator] tryinitFirstPose imu buf %ld\n", imubuf.size());
        initFirstPoseFlag = tryinitFirstPose(frame);
        if (!initFirstPoseFlag) {
            return;
        }
    }

    while (!imubuf.avaiable(frame.stamp + state.td)) {
        //Wait for IMU
        usleep(2000);
        printf("[D2VINS::D2Estimator] wait for imu...\n");
    }
    
}

}