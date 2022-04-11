#pragma once

#include <ros/ros.h>
#include <swarm_msgs/ImageDescriptor.h>
#include <swarm_msgs/LoopEdge.h>
#include <string>
#include <lcm/lcm-cpp.hpp>
#include <swarm_msgs/ImageDescriptor_t.hpp>
#include "swarm_loop/loop_defines.h"
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include <functional>
#include <set>
#include <swarm_msgs/ImageDescriptorHeader_t.hpp>
#include <swarm_msgs/LandmarkDescriptor_t.hpp>
#include <swarm_msgs/FisheyeFrameDescriptor_t.hpp>
#include <mutex>

using namespace swarm_msgs;

class LoopNet {
    lcm::LCM lcm;

    std::set<int64_t> sent_message;

    double recv_period;

    std::mutex recv_lock;

    bool send_img;
    bool send_whole_img_desc;

public:
    LoopNet(std::string _lcm_uri, bool _send_img, bool _send_whole_img_desc, double _recv_period = 0.5):
        lcm(_lcm_uri), send_img(_send_img), send_whole_img_desc(_send_whole_img_desc), recv_period(_recv_period) {
        this->setup_network(_lcm_uri);
        msg_recv_rate_callback = [&](const int, float) {};
    }

    std::map<int64_t, ImageDescriptor_t> received_images;
    std::map<int64_t, double> msg_recv_last_time;

    std::map<int64_t, double> msg_header_recv_time;
    std::map<int64_t, double> frame_header_recv_time;
    
    std::set<int64_t> active_receving_msg;
    std::set<int64_t> active_receving_frames;
    std::set<int64_t> blacklist;
    std::map<int64_t, FisheyeFrameDescriptor_t> received_frames;

    std::function<void(const FisheyeFrameDescriptor_t &)> frame_desc_callback;
    std::function<void(const LoopEdge_t &)> loopconn_callback;
    std::function<void(const int, float)> msg_recv_rate_callback;

    void image_desc_callback(const ImageDescriptor_t & image);
    void setup_network(std::string _lcm_uri);
    void broadcast_img_desc(ImageDescriptor_t & img_des);
    void broadcast_fisheye_desc(FisheyeFrameDescriptor_t & fisheye_desc);

    void broadcast_loop_connection(swarm_msgs::LoopEdge & loop_conn);

    void on_loop_connection_recevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const LoopEdge_t* msg);

    void on_img_desc_recevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const ImageDescriptor_t* msg);

    void on_img_desc_header_recevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const ImageDescriptorHeader_t* msg);

    void on_landmark_recevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const LandmarkDescriptor_t* msg);

    int lcm_handle() {
        return lcm.handle();
    }

    void update_recv_img_desc_ts(int64_t id, bool is_header=false);

    void scan_recv_packets();

    bool msg_blocked(int64_t _id) {
        return blacklist.find(_id) != blacklist.end() || sent_message.find(_id) != sent_message.end();
    }
};
