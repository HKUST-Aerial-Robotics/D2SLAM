#include <d2frontend/d2frontend.h>
#include <d2common/d2frontend_types.h>
#include "sensor_msgs/Imu.h"
#include "estimator/d2estimator.hpp"
#include <mutex>
#include <queue>

#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}

using namespace D2VINS;
using namespace D2Common;

class D2VINSNode :  public D2FrontEnd::D2Frontend
{
    D2Estimator estimator;
    ros::Subscriber imu_sub;
    int frame_count = 0;
    std::queue<D2Common::VisualImageDescArray> viokf_queue;
    std::mutex queue_lock;
    ros::Timer estimator_timer;
protected:
    virtual void frameCallback(const D2Common::VisualImageDescArray & viokf) override {
        if (frame_count % params->frame_step == 0) {
            queue_lock.lock();
            viokf_queue.emplace(viokf);
            queue_lock.unlock();
        }
        frame_count ++;
    };

    void timerCallback(const ros::TimerEvent & event) {
        if (!viokf_queue.empty()) {
            if (viokf_queue.size() > params->warn_pending_frames) {
                ROS_WARN("[D2VINS] Low efficient on D2VINS::estimator pending frames: %d", viokf_queue.size());
            }
            queue_lock.lock();
            D2Common::VisualImageDescArray viokf = viokf_queue.front();
            viokf_queue.pop();
            queue_lock.unlock();
            estimator.inputImage(viokf);
        }
    };

    virtual void imuCallback(const sensor_msgs::Imu & imu) {
        IMUData data(imu);
        data.dt = 1.0/params->IMU_FREQ; //TODO
        estimator.inputImu(data);
    }

public:
    D2VINSNode(ros::NodeHandle & nh) {
        initParams(nh);
        Init(nh);
        estimator.init(nh);
        imu_sub  = nh.subscribe(params->imu_topic, 1, &D2VINSNode::imuCallback, this, ros::TransportHints().tcpNoDelay());
        estimator_timer = nh.createTimer(ros::Duration(1.0/params->estimator_timer_freq), &D2VINSNode::timerCallback, this);
        ROS_INFO("D2VINS node initialized. Ready to start.");
    }
};

int main(int argc, char **argv)
{
    cv::setNumThreads(1);
    ros::init(argc, argv, "d2vins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    D2VINSNode d2vins(n);
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    return 0;
}

