#include "d2frontend/d2frontend.h"

class D2FrontendNode : public D2FrontEnd::D2Frontend {
 public:
  D2FrontendNode(ros::NodeHandle &nh) { Init(nh); }
};

int main(int argc, char **argv) {
  cv::setNumThreads(1);
  ros::init(argc, argv, "d2frontend");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);

  D2FrontendNode frontend(n);
  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();
  return 0;
}
