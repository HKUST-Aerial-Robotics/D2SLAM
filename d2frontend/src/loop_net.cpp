#include <d2frontend/loop_net.h>
#include <time.h> 
#include "d2frontend/loop_detector.h"
#include <swarm_msgs/lcm_gen/LandmarkDescriptorPacket_t.hpp>

namespace D2FrontEnd {
void LoopNet::setupNetwork(std::string _lcm_uri) {
    if (!lcm.good()) {
        ROS_ERROR("LCM %s failed", _lcm_uri.c_str());
        exit(-1);
    }
    lcm.subscribe("VIOKF_HEADER", &LoopNet::onImgDescHeaderRecevied, this);
    lcm.subscribe("VIOKF_LANDMARKS", &LoopNet::onLandmarkRecevied, this);
    lcm.subscribe("VIOKF_IMG_ARRAY", &LoopNet::onImgArrayRecevied, this);
    lcm.subscribe("SWARM_LOOP_CONN", &LoopNet::onLoopConnectionRecevied, this);

    srand((unsigned)time(NULL)); 
    msg_recv_rate_callback = [&](int drone_id, float rate) {};
}

void LoopNet::broadcastVisualImageDescArray(VisualImageDescArray & image_array, bool force_features) {
    bool need_send_features = force_features || !params->lazy_broadcast_keyframe; //TODO: need to consider for D2VINS.
    ImageArrayDescriptor_t fisheye_desc = image_array.toLCM(need_send_features, compress_int8_desc);
    // printf("[LoopNet@%d] broadcast image array: %ld size %d\n", params->self_id, fisheye_desc.frame_id, fisheye_desc.getEncodedSize());
    if (send_whole_img_desc) {
        sent_message.insert(fisheye_desc.msg_id);
        lcm.publish("VIOKF_IMG_ARRAY", &fisheye_desc);
        if (params->print_network_status) {
            int feature_num = fisheye_desc.landmark_num;
            int byte_sent = fisheye_desc.getEncodedSize();
            sum_byte_sent+= byte_sent;
            count_img_desc_sent ++;
            sum_features += feature_num;
            printf("[SWARM_LOOP](%d) BD KF %d LM: %d size %d avgsize %.0f sumkB %.0f avgLM %.0f force_features:%d\n", count_img_desc_sent,
                fisheye_desc.frame_id, feature_num, byte_sent, ceil(sum_byte_sent/count_img_desc_sent), sum_byte_sent/1000, ceil(sum_features/count_img_desc_sent), force_features);
        }
    } else {
        for (auto & img : fisheye_desc.images) {
            if (img.landmark_num > 0) {
                img.header.is_keyframe = fisheye_desc.is_keyframe;
                broadcastImgDesc(img, fisheye_desc.sld_win_status, need_send_features);
            }
        }
    }
}

void LoopNet::broadcastImgDesc(ImageDescriptor_t & img_des, const SlidingWindow_t & sld_status, bool need_send_features) {
    int64_t msg_id = rand() + img_des.header.timestamp.nsec;
    img_des.header.msg_id = msg_id;
    sent_message.insert(img_des.header.msg_id);

    int byte_sent = 0;
    int feature_num = img_des.landmark_num;

    ImageDescriptorHeader_t & img_desc_header = img_des.header;
    img_desc_header.sld_win_status = sld_status;
    img_desc_header.feature_num = feature_num;

    byte_sent += img_desc_header.getEncodedSize();
    lcm.publish("VIOKF_HEADER", &img_desc_header);
    // printf("[LoopNet] Header id %ld msg_id %ld desc_size %ld:%ld\n", img_desc_header.frame_id, img_desc_header.msg_id, 
    //     img_desc_header.image_desc_size_int8, img_desc_header.image_desc_size);
    if (need_send_features) {
        LandmarkDescriptorPacket_t * lm_pack = new LandmarkDescriptorPacket_t();
        lm_pack->desc_len = 0;
        lm_pack->desc_len_int8 = 0;
        for (size_t i = 0; i < img_des.landmark_num; i++ ) {
            if (img_des.landmarks[i].type == LandmarkType::SuperPointLandmark) {
                lm_pack->landmarks.emplace_back(img_des.landmarks[i].compact);
                if (img_des.landmark_descriptor_int8.size() > 0) {
                    lm_pack->desc_len_int8 += params->superpoint_dims;
                    lm_pack->landmark_descriptor_int8.insert(lm_pack->landmark_descriptor_int8.end(), 
                        img_des.landmark_descriptor_int8.data() + i *params->superpoint_dims, 
                        img_des.landmark_descriptor_int8.data() + (i+1)*params->superpoint_dims);
                    lm_pack->desc_len = 0;
                } else {
                    lm_pack->desc_len += params->superpoint_dims;
                    lm_pack->landmark_descriptor.insert(lm_pack->landmark_descriptor.end(), 
                        img_des.landmark_descriptor.data() + i *params->superpoint_dims, 
                        img_des.landmark_descriptor.data() + (i+1)*params->superpoint_dims);
                    lm_pack->desc_len_int8 = 0;
                }

                if (lm_pack->landmarks.size() > pack_landmark_num || i == img_des.landmark_num - 1) {
                    lm_pack->msg_id = rand() + img_des.header.timestamp.nsec;
                    lm_pack->header_id = img_des.header.msg_id;
                    lm_pack->landmark_num = lm_pack->landmarks.size();
                    sent_message.insert(msg_id);
                    byte_sent += lm_pack->getEncodedSize();
                    if (params->print_network_status) {
                        // printf("[LoopNet] BD LMPack LM size %d(%dx%d) superpoint_dims %d lm.landmark_descriptor_int8 %d\n", lm_pack->getEncodedSize(), 
                        //     lm_pack->landmarks.size(), lm_pack->landmarks[0].getEncodedSize(), params->superpoint_dims, lm_pack->landmark_descriptor_int8.size());
                    }
                    lcm.publish("VIOKF_LANDMARKS", lm_pack);
                    delete lm_pack;
                    if (i != img_des.landmark_num - 1) {
                        lm_pack = new LandmarkDescriptorPacket_t();
                        lm_pack->desc_len = 0;
                        lm_pack->desc_len_int8 = 0;
                    }
                }
            }
        }
    }

    sum_byte_sent+= byte_sent;
    sum_features+=feature_num;
    count_img_desc_sent ++;
    if (params->print_network_status) {
        printf("[SWARM_LOOP](%d) BD KF %d@%d LM: %d size %d header %d avgsize %.0f sumkB %.0f avgLM %.0f need_send_features: %d\n", count_img_desc_sent,
            img_desc_header.frame_id,  img_desc_header.camera_index, feature_num, byte_sent, img_desc_header.getEncodedSize(), 
            ceil(sum_byte_sent/count_img_desc_sent), sum_byte_sent/1000, ceil(sum_features/count_img_desc_sent), need_send_features);
    }
}

void LoopNet::broadcastLoopConnection(swarm_msgs::LoopEdge & loop_conn) {
    auto _loop_conn = toLCMLoopEdge(loop_conn);
    sent_message.insert(_loop_conn.id);
    lcm.publish("SWARM_LOOP_CONN", &_loop_conn);
}

void LoopNet::onImgArrayRecevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const ImageArrayDescriptor_t* msg) {
    std::lock_guard<std::recursive_mutex> Guard(recv_lock);
    if (sent_message.find(msg->msg_id) == sent_message.end()) {
        frame_desc_callback(*msg);
    }
}

void LoopNet::onLandmarkRecevied(const lcm::ReceiveBuffer* rbuf,
        const std::string& chan, 
        const LandmarkDescriptorPacket_t* msg) {
    std::lock_guard<std::recursive_mutex> Guard(recv_lock);
    if(msgBlocked(msg->header_id)) {
        return;
    }
    updateRecvImgDescTs(msg->header_id, false);
    if (received_images.find(msg->header_id) == received_images.end()) {
        //May happen when the image is not received yet
        return;
    }

    auto & tmp = received_images[msg->header_id];
    for (auto landmark: msg->landmarks) {
        tmp.landmarks.emplace_back(Landmark_t());
        auto & new_lm = tmp.landmarks.back();
        new_lm.compact = landmark;
        new_lm.camera_id = tmp.header.camera_id;
        new_lm.camera_index = tmp.header.camera_index;
        new_lm.frame_id = tmp.header.frame_id;
        new_lm.drone_id = tmp.header.drone_id;
        new_lm.type = LandmarkType::SuperPointLandmark;
        new_lm.timestamp = tmp.header.timestamp;
        new_lm.cur_td = tmp.header.cur_td;
    }

    if (msg->landmark_descriptor_int8.size() > 0) {
        tmp.landmark_descriptor_int8.insert(tmp.landmark_descriptor_int8.end(), msg->landmark_descriptor_int8.begin(), msg->landmark_descriptor_int8.end());
        tmp.landmark_descriptor_size_int8 = tmp.landmark_descriptor_int8.size();
        tmp.landmark_descriptor_size = 0;
        // printf("Adding landmark_descriptor_int8 %ld to image_desc %ld: %ld\n", msg->landmark_descriptor_int8.size(), tmp.frame_id, tmp.landmark_descriptor_size_int8);
    } else {
        tmp.landmark_descriptor.insert(tmp.landmark_descriptor.end(), msg->landmark_descriptor.begin(), msg->landmark_descriptor.end());
        tmp.landmark_descriptor_size = tmp.landmark_descriptor.size();
        tmp.landmark_descriptor_size_int8 = 0;
    }
    scanRecvPackets();
}

void LoopNet::processRecvImageDesc(const ImageDescriptor_t & image, const SlidingWindow_t & sld_win_status) {
    std::lock_guard<std::recursive_mutex> Guard(recv_lock);
    int64_t frame_id = image.header.frame_id;
    if (received_framearrays.find(frame_id) == received_framearrays.end()) {
        ImageArrayDescriptor_t frame_desc;
        if (params->camera_configuration == CameraConfig::STEREO_FISHEYE) {
            frame_desc.image_num = 4;
        } else if (params->camera_configuration == CameraConfig::STEREO_PINHOLE) {
            frame_desc.image_num = 2;
        } else if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
            frame_desc.image_num = 1;
        } else if (params->camera_configuration == CameraConfig::FOURCORNER_FISHEYE) {
            frame_desc.image_num = 4;
        }

        frame_desc.timestamp = image.header.timestamp;
        for (size_t i = 0; i < frame_desc.image_num; i ++) {
            if (i != image.header.camera_index) {
                auto img_desc = generate_null_img_desc();           
                frame_desc.images.push_back(img_desc);
            } else {
                frame_desc.images.push_back(image);
            }
        }

        frame_desc.msg_id = image.header.frame_id;
        frame_desc.pose_drone = image.header.pose_drone;
        frame_desc.landmark_num = 0;
        frame_desc.drone_id = image.header.drone_id;
        frame_desc.frame_id = image.header.frame_id;
        frame_desc.is_lazy_frame = image.header.is_lazy_frame;
        frame_desc.matched_frame = image.header.matched_frame;
        frame_desc.matched_drone = image.header.matched_drone;
        frame_desc.sld_win_status = sld_win_status;
        frame_desc.reference_frame_id = image.header.reference_frame_id;
        frame_desc.Ba.x = 0;
        frame_desc.Ba.y = 0;
        frame_desc.Ba.z = 0;
        frame_desc.Bg.x = 0;
        frame_desc.Bg.y = 0;
        frame_desc.Bg.z = 0;
        frame_desc.is_keyframe = image.header.is_keyframe;
        frame_desc.cur_td = image.header.cur_td;
        received_framearrays[image.header.frame_id] = frame_desc;
        frame_header_recv_time[image.header.frame_id] = msg_header_recv_time[image.header.msg_id];
        active_receving_frames.insert(image.header.frame_id);
        if (params->print_network_status) {
            printf("[LoopNet::processRecvImageDesc] Create frame %dc%d from D%d \n", frame_id, 
                    image.header.camera_index, frame_desc.drone_id);
        }
    } else {
        auto & frame_desc = received_framearrays[frame_id];
        frame_desc.images[image.header.camera_index] = image;
        if (params->print_network_status) {
            printf("[LoopNet::processRecvImageDesc] Adding subframe %dc%d D%d to current\n", frame_id, 
                    image.header.camera_index, frame_desc.drone_id);
        }
    }
}

void LoopNet::onLoopConnectionRecevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const LoopEdge_t* msg) {

    if (sent_message.find(msg->id) != sent_message.end()) {
        // ROS_INFO("Receive self sent Loop message");
        return;
    }
    ROS_INFO("Received Loop %d->%d from LCM!!!", msg->drone_id_a, msg->drone_id_b);    
    loopconn_callback(*msg);
}


void LoopNet::onImgDescHeaderRecevied(const lcm::ReceiveBuffer* rbuf,
    const std::string& chan, 
    const ImageDescriptorHeader_t* msg) {
    std::lock_guard<std::recursive_mutex> Guard(recv_lock);

    if(msgBlocked(msg->msg_id)) {
        return;
    }
    if (msg->matched_drone >= 0 && msg->matched_drone != params->self_id) {
        printf("[LoopNet@%d] Received image desc from %d but matched to %d. Skip\n", params->self_id, msg->drone_id, msg->matched_drone);
        return;
    } else if (msg->matched_drone >= 0) {
        printf("[LoopNet@%d] Received image desc from %d matched to frame %ld\n", params->self_id, msg->drone_id, msg->frame_id);
    }

    if (params->print_network_status) {
        printf("[LoopNet]ImageDescriptorHeader %ldc%ld from D%d msg_id %ld: feature num %d gdesc %d:%d\n", msg->frame_id, msg->camera_index, msg->drone_id, 
            msg->msg_id, msg->feature_num, msg->image_desc_size_int8, msg->image_desc_size);
    }
    updateRecvImgDescTs(msg->msg_id, true);

    if (received_images.find(msg->msg_id) == received_images.end()) {
        ImageDescriptor_t tmp;
        tmp.landmark_descriptor_size_int8 = 0;
        tmp.landmark_descriptor_size = 0;
        tmp.landmark_scores_size = 0;
        active_receving_msg.insert(msg->msg_id);
        received_images[msg->msg_id] = tmp; 
    }
    received_sld_win_status[msg->msg_id] = msg->sld_win_status;
    received_images[msg->msg_id].header = *msg;
    received_images[msg->msg_id].landmark_num = msg->feature_num;
}

void LoopNet::scanRecvPackets() {
    std::lock_guard<std::recursive_mutex> Guard(recv_lock);
    double tnow = ros::Time::now().toSec();
    std::vector<int64_t> finish_recv;
    static double sum_feature_num = 0;
    static double sum_feature_num_all = 0;
    static int sum_packets = 0;
    for (auto msg_id : active_receving_msg) {
        auto & _frame = received_images[msg_id];
        if (tnow - msg_header_recv_time[msg_id] > recv_period ||
            _frame.landmark_num == _frame.landmarks.size()) {
            sum_feature_num_all+=_frame.landmark_num;
            sum_feature_num+=_frame.landmarks.size();
            float cur_recv_rate = ((float)_frame.landmarks.size())/((float) _frame.landmark_num);
            if (params->print_network_status) {
                printf("[LoopNet](%d) frame %ldc%d from D%d msg_id %ld , LM %d/%d recv duration: %.3fs recv_rate avg %.1f cur %.1f", 
                    sum_packets, _frame.header.frame_id, _frame.header.camera_index, _frame.header.drone_id, msg_id, _frame.landmarks.size(), _frame.landmark_num,
                    tnow - msg_header_recv_time[msg_id], sum_feature_num/sum_feature_num_all*100, cur_recv_rate*100);
                printf(" gdesc_size %d/%d lm_desc_size %d/%d\n",  _frame.header.image_desc_size_int8, _frame.header.image_desc_size,
                    _frame.landmark_descriptor_size_int8,  _frame.landmark_descriptor_size);
            }
            _frame.landmark_num = _frame.landmarks.size();
            finish_recv.push_back(msg_id);
            images_finish_recv.insert(msg_id);

            sum_packets += 1;
            msg_recv_rate_callback(_frame.header.drone_id, cur_recv_rate);
        }
    }

    for (auto msg_id : finish_recv) {
        blacklist.insert(msg_id);
        active_receving_msg.erase(msg_id);
    }


    for (auto _id : finish_recv) {
        auto & msg = received_images[_id];
        //Processed recevied message
        msg.landmark_num = msg.landmarks.size();
        if (msg.landmarks.size() > 0) {
            this->processRecvImageDesc(msg, received_sld_win_status[_id]);
        }
        received_images.erase(_id);
        received_sld_win_status.erase(_id);
    }

    //Scan finish image array
    std::vector<int64_t> finish_recv_frames;
    for (auto frame_id : active_receving_frames) {
        int count_images = 0;
        auto & frame_desc = received_framearrays[frame_id];
        for (size_t i = 0; i < frame_desc.images.size(); i++) {
            if (frame_desc.images[i].landmark_num > 0 && 
                    images_finish_recv.find(frame_desc.images[i].header.msg_id) != images_finish_recv.end()) {
                count_images ++;
            }
        }

        if(frame_header_recv_time.find(frame_id) != frame_header_recv_time.end() &&
                (tnow - frame_header_recv_time[frame_id] > recv_period || count_images >= params->loopdetectorconfig->MIN_DIRECTION_LOOP)) {
            finish_recv_frames.push_back(frame_id);
        }
    }

    for (auto & frame_id :finish_recv_frames) {
        auto & frame_desc = received_framearrays[frame_id];
        active_receving_frames.erase(frame_id);

        frame_desc.landmark_num = 0;
        for (size_t i = 0; i < frame_desc.images.size(); i ++) {
            frame_desc.landmark_num += frame_desc.images[i].landmark_num;
            if (params->print_network_status) {
                printf("[LoopNet::finishRecvArray] frame %ldc%d from D%d, LM %ld/%d gdesc %d %d\n", 
                        frame_desc.images[i].header.frame_id, frame_desc.images[i].header.camera_index, 
                        frame_desc.images[i].header.drone_id, frame_desc.images[i].landmarks.size(), frame_desc.images[i].landmark_num,
                        frame_desc.images[i].header.image_desc_size_int8, frame_desc.images[i].header.image_desc_size);
            }
            images_finish_recv.erase(frame_desc.images[i].header.msg_id);
            if (frame_desc.images[i].header.frame_id < 0) {
                //Has empty header frame
                return;
            }
        }

        if (params->print_network_status) {
            printf("[LoopNet@%d] Recv frame %ld: %d images from drone %d, landmark %d\n", params->self_id, 
                    frame_desc.frame_id, frame_desc.images.size(), frame_desc.drone_id, frame_desc.landmark_num);
        }

        frame_desc_callback(frame_desc);
        received_framearrays.erase(frame_id);
    }
}

void LoopNet::updateRecvImgDescTs(int64_t id, bool is_header) {
    if(is_header) {
        msg_header_recv_time[id] = ros::Time::now().toSec();
    }
    msg_recv_last_time[id] = ros::Time::now().toSec();
}
}