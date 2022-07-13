#include "d2comm.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "d2comm");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    D2Comm::D2Comm d2comm;
    d2comm.init(n);
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    return 0;
}