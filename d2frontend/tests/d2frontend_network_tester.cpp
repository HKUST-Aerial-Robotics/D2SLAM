#include <thread>

#include "d2frontend/loop_net.h"
#include "swarm_msgs/swarm_lcm_converter.hpp"
#include "swarmcomm_msgs/drone_network_status.h"

using namespace D2FrontEnd;
class SwarmNetworkTester {
  LoopNet loopnet;
  ros::Publisher drone_status_pub;
  ros::Timer timer;
  std::thread th;

 public:
  int self_id = -1;
  SwarmNetworkTester(ros::NodeHandle &nh)
      : loopnet("udpm://224.0.0.251:7667?ttl=255", false, false) {
    nh.param<int>("self_id", self_id, -1);
    loopnet.msg_recv_rate_callback = [&](int drone_id, float rate) {
      this->receive_rate_callback(drone_id, rate);
    };

    loopnet.frame_desc_callback = [&](const VisualImageDescArray &) {};

    drone_status_pub = nh.advertise<swarmcomm_msgs::drone_network_status>(
        "/swarm_loop/drone_network_status", 10);
    timer = nh.createTimer(ros::Duration(1.0),
                           &SwarmNetworkTester::timerCallback, this);
    th = std::thread([&] {
      while (0 == loopnet.lcmHandle()) {
      }
    });
  }

  void receive_rate_callback(int drone_id, float rate) {
    swarmcomm_msgs::drone_network_status status;
    status.header.stamp = ros::Time::now();
    status.drone_id = drone_id;
    status.active = true;
    status.quality = rate;
    status.bandwidth = -1;
    status.hops = -1;
    drone_status_pub.publish(status);
  }

  void timerCallback(const ros::TimerEvent &e) {
    static int count = 0;
    ImageDescriptor_t dummy_desc;
    dummy_desc.header.timestamp = toLCMTime(ros::Time::now());
    dummy_desc.header.drone_id = self_id;
    dummy_desc.header.msg_id = count + self_id * 1000000;
    dummy_desc.landmark_num = 200;
    dummy_desc.landmark_descriptor.resize(200 * 64);
    dummy_desc.landmark_descriptor_size = dummy_desc.landmark_descriptor.size();
    dummy_desc.header.image_desc.resize(4096);
    dummy_desc.header.image_desc_size = 4096;
    dummy_desc.image_size = 0;
    dummy_desc.landmarks.resize(dummy_desc.landmark_num);
    for (auto &lm : dummy_desc.landmarks) {
      lm.compact.flag = 1;
    }
    loopnet.broadcastImgDesc(dummy_desc, SlidingWindow_t());
    count++;
  }
};

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "swarm_network_tester");
  ros::NodeHandle nh("swarm_network_tester");
  SwarmNetworkTester tester(nh);
  ROS_INFO("swarm network tester at %d\nIniting\n", tester.self_id);
  ros::spin();
}