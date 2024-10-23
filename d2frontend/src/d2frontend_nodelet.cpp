#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include "d2frontend/d2frontend.h"

#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace D2FrontEnd {
class D2FrontendNode : public nodelet::Nodelet, public D2Frontend {
 public:
  D2FrontendNode() {}

 private:
  virtual void onInit() override {
    ros::NodeHandle& n = getMTPrivateNodeHandle();
    Init(n);
    cv::setNumThreads(1);
  }
};
}  // namespace D2FrontEnd

PLUGINLIB_EXPORT_CLASS(D2FrontEnd::D2FrontendNode, nodelet::Nodelet);