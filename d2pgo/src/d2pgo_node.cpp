#include <ros/ros.h>
#include "d2pgo.h"

namespace D2PGO {
class D2PGONode {
    D2PGO * pgo = nullptr;
protected:
    void Init(ros::NodeHandle & nh) {
        D2PGOConfig config;
        pgo = new D2PGO(config);
    }
public:
    D2PGONode(ros::NodeHandle & nh) {
        Init(nh);
    }
};
}

int main(int argc, char **argv)
{
    cv::setNumThreads(1);
    ros::init(argc, argv, "d2pgo");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    D2PGO::D2PGONode d2pgonode(n);
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    return 0;
}