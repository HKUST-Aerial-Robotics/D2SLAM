#include <d2frontend/d2frontend.h>
#include <d2frontend/d2frontend_types.h>
#include "sensor_msgs/Imu.h"
#include "estimator/d2estimator.hpp"

using namespace D2VINS;
using namespace D2Frontend;
class D2VINSNode :  public D2Frontend
{
    D2Estimator estimator;
    ros::Subscriber imu_sub;

protected:
    virtual void frameCallback(const VisualImageDescArray & viokf) override {
        VisualImageDescArray _viokf = viokf; //Here we do not need to copy desc.
        estimator.inputImage(_viokf);
    };

    virtual void imu_callback(const sensor_msgs::Imu & imu) {
        IMUData data(imu);
        estimator.inputImu(data);
    }

public:
    D2VINSNode(ros::NodeHandle & nh) {
        imu_sub  = nh.subscribe("imu", 1, &D2VINSNode::imu_callback, this, ros::TransportHints().tcpNoDelay());
        Init(nh);
    }
    
};

int main(int argc, char **argv)
{
    cv::setNumThreads(1);
    ros::init(argc, argv, "d2vins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    D2VINSNode d2vins(n);
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    return 0;
}

