#include "ros/ros.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "d2vins-msckf");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    std::cout << "Hello world!" << std::endl;
}