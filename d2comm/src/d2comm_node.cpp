#include "d2comm.h"

int main(int argc, char **argv) {
  printf("D2Comm starting....\n");
  ros::init(argc, argv, "d2comm");
  ros::NodeHandle n("~");
  D2Comm::D2Comm d2comm;
  d2comm.init(n);
  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();
  return 0;
}