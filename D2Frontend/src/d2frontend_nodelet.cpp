#include "d2frontend/d2frontend.h"
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#define BACKWARD_HAS_DW 1
#include <backward.hpp>
namespace backward
{
    backward::SignalHandling sh;
}

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace d2slam
{
    class D2SLAMNode : public nodelet::Nodelet, public D2Frontend
    {
        public:
            SwarmLoopNode() {}
        private:
            virtual void onInit() override
            {
                ros::NodeHandle & n = getMTPrivateNodeHandle();
                Init(n);
                cv::setNumThreads(1);
            }
    };
    PLUGINLIB_EXPORT_CLASS(swarm_localization_pkg::SwarmLoopNode, nodelet::Nodelet);
}
