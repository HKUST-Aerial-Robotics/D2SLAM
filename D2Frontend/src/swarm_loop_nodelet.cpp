#include "swarm_loop/swarm_loop.h"
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

namespace swarm_localization_pkg
{
    class SwarmLoopNode : public nodelet::Nodelet, public SwarmLoop
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
