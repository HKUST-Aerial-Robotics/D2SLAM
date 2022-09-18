#pragma once

#include <ros/ros.h>
#include <swarm_msgs/ImageDescriptor.h>
#include <swarm_msgs/LoopEdge.h>
#include <string>
#include <lcm/lcm-cpp.hpp>
#include "d2frontend/d2frontend_params.h"
#include "d2common/d2frontend_types.h"
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include <functional>
#include <set>
#include <mutex>
#include <thread>

using namespace swarm_msgs;
using namespace D2Common;

namespace D2FrontEnd {
class LoopNet {
    lcm::LCM lcm;

    std::set<int64_t> sent_message;

    double recv_period;

    std::mutex recv_lock;

    bool send_img;
    bool send_whole_img_desc;

    void onLoopConnectionRecevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const LoopEdge_t* msg);

    void onImgArrayRecevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const ImageArrayDescriptor_t* msg);

    void onImgDescHeaderRecevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const ImageDescriptorHeader_t* msg);

    void onLandmarkRecevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const LandmarkDescriptor_t* msg);
    std::map<int64_t, ImageDescriptor_t> received_images;
    std::map<int64_t, double> msg_recv_last_time;

    std::map<int64_t, double> msg_header_recv_time;
    std::map<int64_t, double> frame_header_recv_time;
    
    std::set<int64_t> active_receving_msg;
    std::set<int64_t> active_receving_frames;
    std::set<int64_t> blacklist;
    std::map<int64_t, ImageArrayDescriptor_t> received_frames;


    void setupNetwork(std::string _lcm_uri);
    void imageDescCallback(const ImageDescriptor_t & image);
    void updateRecvImgDescTs(int64_t id, bool is_header=false);
    bool msgBlocked(int64_t _id) {
        return blacklist.find(_id) != blacklist.end() || sent_message.find(_id) != sent_message.end();
    }
public:
    std::function<void(const VisualImageDescArray &)> frame_desc_callback;
    std::function<void(const LoopEdge_t &)> loopconn_callback;
    std::function<void(const int, float)> msg_recv_rate_callback;

    LoopNet(std::string _lcm_uri, bool _send_img, bool _send_whole_img_desc, double _recv_period = 0.5):
        lcm(_lcm_uri), send_img(_send_img), send_whole_img_desc(_send_whole_img_desc), recv_period(_recv_period) {
        this->setupNetwork(_lcm_uri);
        msg_recv_rate_callback = [&](const int, float) {};
    }

    void broadcastLoopConnection(swarm_msgs::LoopEdge & loop_conn);
    void broadcastVisualImageDescArray(VisualImageDescArray & image_array, bool force_features=false);
    void broadcastImgDesc(ImageDescriptor_t & img_des, bool send_feature = true);

    void scanRecvPackets();

    int lcmHandle() {
        return lcm.handle();
    }
};
}
