#include "d2estimator.hpp" 
#include "unistd.h"

namespace D2VINS {
void D2Estimator::inputImu(IMUData data) {
    imu_buf.add(data);
    if (!initFirstPoseFlag) {
        return;
    }
}

bool D2Estimator::tryinitFirstPose() {
    if (imubuf.size() < _config.init_imu_num) {
        return false;
    }
    auto q0 = g2R(imubuf.mean_acc());
    last_pose = Swarm::Pose(q0, Vector3d::Zero());
    printf("[D2VINS::D2Estimator] Init pose with IMU: %s\n", last_pose.tostr().c_str());
}

void D2Estimator::inputImage(VisualImageDescArray & frame) {
    if(!initFirstPoseFlag) {
        initFirstPoseFlag = tryinitFirstPose();
        if (!initFirstPoseFlag) {
            return;
        }
    }

    while (!imubuf.avaiable(frame.stamp.to_sec() + td)) {
        //Wait for IMU
        usleep(2000);
        printf("[D2VINS::D2Estimator] wait for imu...\n");
    }
    
}

}