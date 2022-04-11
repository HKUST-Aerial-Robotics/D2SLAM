#include "swarm_loop/loop_net.h"
#include <time.h> 

void LoopNet::setup_network(std::string _lcm_uri) {
    if (!lcm.good()) {
        ROS_ERROR("LCM %s failed", _lcm_uri.c_str());
        exit(-1);
    }
    lcm.subscribe("SWARM_LOOP_IMG_DES", &LoopNet::on_img_desc_recevied, this);
    lcm.subscribe("SWARM_LOOP_CONN", &LoopNet::on_loop_connection_recevied, this);
    
    lcm.subscribe("VIOKF_HEADER", &LoopNet::on_img_desc_header_recevied, this);
    lcm.subscribe("VIOKF_LANDMARKS", &LoopNet::on_landmark_recevied, this);

    srand((unsigned)time(NULL)); 
    msg_recv_rate_callback = [&](int drone_id, float rate) {};
}


void LoopNet::broadcast_fisheye_desc(FisheyeFrameDescriptor_t & fisheye_desc) {
    //Broadcast Three ImageDesc
    for (auto & img : fisheye_desc.images) {
        if (img.landmark_num > 0)
        broadcast_img_desc(img);
    }
}

void LoopNet::broadcast_img_desc(ImageDescriptor_t & img_des) {
    int64_t msg_id = rand() + img_des.timestamp.nsec;
    img_des.msg_id = msg_id;
    sent_message.insert(img_des.msg_id);
    static double sum_byte_sent = 0;
    static double sum_features = 0;
    static int count_byte_sent = 0;

    int byte_sent = 0;
    if (IS_PC_REPLAY) {
        ROS_INFO("Sending IMG DES Size %d with %d landmarks in PC replay mode.local feature size %d", img_des.getEncodedSize(), img_des.landmark_num, img_des.feature_descriptor_size);
        lcm.publish("SWARM_LOOP_IMG_DES", &img_des);
        return;
    }

    int feature_num = 0;
    for (size_t i = 0; i < img_des.landmark_num; i++ ) {
        if (img_des.landmarks_flag[i] > 0) {
            feature_num ++;
        }
    }


    ImageDescriptorHeader_t img_desc_header;
    img_desc_header.timestamp = img_des.timestamp;
    img_desc_header.drone_id = img_des.drone_id;
    img_desc_header.image_desc = img_des.image_desc;
    img_desc_header.pose_drone = img_des.pose_drone;
    img_desc_header.camera_extrinsic = img_des.camera_extrinsic;
    img_desc_header.prevent_adding_db = img_des.prevent_adding_db;
    img_desc_header.msg_id = img_des.msg_id;
    img_desc_header.frame_id = img_des.frame_id;
    img_desc_header.image_desc_size = img_des.image_desc_size;
    img_desc_header.image_desc = img_des.image_desc;
    img_desc_header.feature_num = feature_num;
    img_desc_header.direction = img_des.direction;

    byte_sent += img_desc_header.getEncodedSize();
    lcm.publish("VIOKF_HEADER", &img_desc_header);
    // printf("header %d", img_desc_header.getEncodedSize());
    for (size_t i = 0; i < img_des.landmark_num; i++ ) {
        if (img_des.landmarks_flag[i] > 0 || SEND_ALL_FEATURES) {
            LandmarkDescriptor_t lm;
            lm.landmark_id = i;
            lm.landmark_2d_norm = img_des.landmarks_2d_norm[i];
            lm.landmark_2d = img_des.landmarks_2d[i];
            lm.landmark_3d = img_des.landmarks_3d[i];
            lm.landmark_flag = img_des.landmarks_flag[i];
            lm.drone_id = img_des.drone_id;
            lm.desc_len = FEATURE_DESC_SIZE;
            lm.feature_descriptor = std::vector<float>(img_des.feature_descriptor.data() + i *FEATURE_DESC_SIZE, 
                img_des.feature_descriptor.data() + (i+1)*FEATURE_DESC_SIZE);
            int64_t msg_id = rand() + img_des.timestamp.nsec;
            sent_message.insert(img_des.msg_id);

            lm.msg_id = msg_id;
            lm.header_id = img_des.msg_id;
            byte_sent += lm.getEncodedSize();

            // if (i == 0) {
            //     printf("lm %d", lm.getEncodedSize());
            // }

            lcm.publish("VIOKF_LANDMARKS", &lm);
        }
    }

    sum_byte_sent+= byte_sent;
    sum_features+=feature_num;
    count_byte_sent ++;

    ROS_INFO("[SWARM_LOOP](%d) BD KF %d LM: %d size %d avgsize %.0f sumkB %.0f avgLM %.0f", count_byte_sent,
            img_desc_header.msg_id, feature_num, byte_sent, ceil(sum_byte_sent/count_byte_sent), sum_byte_sent/1000, ceil(sum_features/count_byte_sent));


    if (send_img || send_whole_img_desc) {
        if (!send_whole_img_desc) {
            ImageDescriptor_t img_desc_new = img_des;
            img_desc_new.feature_descriptor_size = 0;
            img_desc_new.feature_descriptor.clear();
            byte_sent += img_desc_new.getEncodedSize();
            ROS_INFO("Sending IMG DES Size %d with %d landmarks.local feature size %d", img_desc_new.getEncodedSize(), img_desc_new.landmark_num, img_desc_new.feature_descriptor_size);
            lcm.publish("SWARM_LOOP_IMG_DES", &img_desc_new);
        } else {
            byte_sent += img_des.getEncodedSize();
            ROS_INFO("Sending IMG DES Size %d with %d landmarks.local feature size %d", img_des.getEncodedSize(), img_des.landmark_num, img_des.feature_descriptor_size);
            lcm.publish("SWARM_LOOP_IMG_DES", &img_des);
        }
    }
    

    // ROS_INFO("Sent Message KEYFRAME %ld with %d/%d landmarks g_desc %d total %d bytes", msg_id, feature_num,img_des.landmark_num, img_desc_header.image_desc_size, byte_sent);
}

void LoopNet::broadcast_loop_connection(swarm_msgs::LoopEdge & loop_conn) {
    auto _loop_conn = toLCMLoopEdge(loop_conn);

    sent_message.insert(_loop_conn.id);
    lcm.publish("SWARM_LOOP_CONN", &_loop_conn);
}

void LoopNet::on_img_desc_recevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const ImageDescriptor_t* msg) {
    
    if (sent_message.find(msg->msg_id) != sent_message.end()) {
        // ROS_INFO("Receive self sent IMG message");
        return;
    }
    
    ROS_INFO("Received drone %d image from LCM!!!", msg->drone_id);
    this->image_desc_callback(*msg);
}

void LoopNet::image_desc_callback(const ImageDescriptor_t & image){
    int64_t frame_hash = image.msg_id;

    if (received_frames.find(frame_hash) == received_frames.end()) {
        FisheyeFrameDescriptor_t frame_desc;

        frame_desc.image_num = 4;
        frame_desc.timestamp = image.timestamp;
        for (size_t i = 0; i < frame_desc.image_num; i ++) {
            if (i != image.direction) {
                auto img_desc = generate_null_img_desc();           
                frame_desc.images.push_back(img_desc);
            } else {
                frame_desc.images.push_back(image);
            }
        }

        frame_desc.msg_id = image.frame_id;
        frame_desc.pose_drone = image.pose_drone;
        frame_desc.landmark_num = 0;
        frame_desc.drone_id = image.drone_id;
        received_frames[frame_hash] = frame_desc;
        frame_header_recv_time[frame_hash] = msg_header_recv_time[image.msg_id];

        active_receving_frames.insert(frame_hash);
    } else {
        auto & frame_desc = received_frames[frame_hash];
        ROS_INFO("Adding image to frame %d from drone_id %d direction %d", frame_hash, frame_desc.drone_id, image.direction);
        frame_desc.images[image.direction] = image;
    }
}

void LoopNet::on_loop_connection_recevied(const lcm::ReceiveBuffer* rbuf,
                const std::string& chan, 
                const LoopEdge_t* msg) {

    if (sent_message.find(msg->id) != sent_message.end()) {
        // ROS_INFO("Receive self sent Loop message");
        return;
    }
    ROS_INFO("Received Loop %d->%d from LCM!!!", msg->drone_id_a, msg->drone_id_b);    
    loopconn_callback(*msg);
}


void LoopNet::on_img_desc_header_recevied(const lcm::ReceiveBuffer* rbuf,
    const std::string& chan, 
    const ImageDescriptorHeader_t* msg) {

    if(msg_blocked(msg->msg_id)) {
        return;
    }

    recv_lock.lock();

    ROS_INFO("ImageDescriptorHeader from drone (%d): msg_id: %ld feature num %d", msg->drone_id, msg->msg_id, msg->feature_num);
    update_recv_img_desc_ts(msg->msg_id, true);

    if (received_images.find(msg->msg_id) == received_images.end()) {
        ImageDescriptor_t tmp;
        received_images[msg->msg_id] = tmp; 
        active_receving_msg.insert(msg->msg_id);
    }

    auto & tmp = received_images[msg->msg_id];
    tmp.timestamp = msg->timestamp;
    tmp.drone_id = msg->drone_id;
    tmp.image_desc_size = msg->image_desc_size;
    tmp.image_desc = msg->image_desc;
    tmp.pose_drone = msg->pose_drone;
    tmp.camera_extrinsic = msg->camera_extrinsic;
    tmp.landmark_num = msg->feature_num;
    tmp.frame_id = msg->frame_id;
    tmp.msg_id = msg->msg_id;
    tmp.prevent_adding_db = msg->prevent_adding_db;
    tmp.direction = msg->direction;
    tmp.feature_descriptor_size = 0;

    recv_lock.unlock();
}

void LoopNet::scan_recv_packets() {
    double tnow = ros::Time::now().toSec();
    std::vector<int64_t> finish_recv;
    recv_lock.lock();
    static double sum_feature_num = 0;
    static double sum_feature_num_all = 0;
    static int sum_packets = 0;
    for (auto msg_id : active_receving_msg) {
        if (tnow - msg_header_recv_time[msg_id] > recv_period ||
            received_images[msg_id].landmark_num == received_images[msg_id].landmarks_2d.size()) {
            sum_feature_num_all+=received_images[msg_id].landmark_num;
            sum_feature_num+=received_images[msg_id].landmarks_2d.size();
            float cur_recv_rate = ((float)received_images[msg_id].landmarks_2d.size())/((float) received_images[msg_id].landmark_num);
            ROS_INFO("[SWAMR_LOOP] Frame %d id %ld from drone %d, Feature %ld/%d recv_rate %.1f cur %.1f feature_desc_size %ld(%ld)", 
                sum_packets,
                msg_id, received_images[msg_id].drone_id, received_images[msg_id].landmarks_2d.size(), received_images[msg_id].landmark_num,
                sum_feature_num/sum_feature_num_all*100,
                cur_recv_rate*100,
                received_images[msg_id].feature_descriptor.size(), received_images[msg_id].feature_descriptor_size);
            received_images[msg_id].landmark_num = received_images[msg_id].landmarks_2d.size();
            finish_recv.push_back(msg_id);

            sum_packets += 1;
            msg_recv_rate_callback(received_images[msg_id].drone_id, cur_recv_rate);
        }
    }

    for (auto msg_id : finish_recv) {
        blacklist.insert(msg_id);
        active_receving_msg.erase(msg_id);
    }


    for (auto _id : finish_recv) {
        auto & msg = received_images[_id];
        //Processed recevied message
        msg.landmark_num = msg.landmarks_2d.size();
        if (msg.landmarks_2d.size() > 0) {
            this->image_desc_callback(msg);
        }
        received_images.erase(_id);
    }

    std::vector<int64_t> finish_recv_frames;
    for (auto frame_hash : active_receving_frames) {
        int count_images = 0;
        auto & frame_desc = received_frames[frame_hash];
        for (size_t i = 0; i < frame_desc.images.size(); i++) {
            if (frame_desc.images[i].landmark_num > 0) {
                count_images++;
            }
        }

        if(tnow - frame_header_recv_time[frame_hash] > 2.0*recv_period  || count_images >= MIN_DIRECTION_LOOP) {
            finish_recv_frames.push_back(frame_hash);
        }
    }

    for (auto & frame_hash :finish_recv_frames) {
        auto & frame_desc = received_frames[frame_hash];
        active_receving_frames.erase(frame_hash);

        frame_desc.landmark_num = 0;
        for (size_t i = 0; i < frame_desc.images.size(); i ++) {
            frame_desc.landmark_num += frame_desc.images[i].landmark_num;
        }

        ROS_INFO("[SWAMR_LOOP] FFrame contains of %d images from drone %d, landmark %d", frame_desc.images.size(), frame_desc.drone_id, frame_desc.landmark_num );

        frame_desc_callback(frame_desc);
        received_frames.erase(frame_hash);
    }
    recv_lock.unlock();
}

void LoopNet::on_landmark_recevied(const lcm::ReceiveBuffer* rbuf,
    const std::string& chan, 
    const LandmarkDescriptor_t* msg) {
    if(msg_blocked(msg->header_id)) {
        return;
    }
    recv_lock.lock();
    update_recv_img_desc_ts(msg->header_id, false);
    if (received_images.find(msg->header_id) == received_images.end()) {
        ImageDescriptor_t tmp;
        received_images[msg->header_id] = tmp; 
    }

    auto & tmp = received_images[msg->header_id];
    tmp.landmarks_2d_norm.push_back(msg->landmark_2d_norm);
    tmp.landmarks_2d.push_back(msg->landmark_2d);
    tmp.landmarks_3d.push_back(msg->landmark_3d);
    tmp.landmarks_flag.push_back(msg->landmark_flag);
    tmp.feature_descriptor.insert(tmp.feature_descriptor.end(),
        msg->feature_descriptor.begin(),
        msg->feature_descriptor.begin()+FEATURE_DESC_SIZE
    );
    tmp.feature_descriptor_size = tmp.feature_descriptor.size();
    recv_lock.unlock();
    
    scan_recv_packets();
}


void LoopNet::update_recv_img_desc_ts(int64_t id, bool is_header) {
    if(is_header) {
        msg_header_recv_time[id] = ros::Time::now().toSec();
    }
    msg_recv_last_time[id] = ros::Time::now().toSec();
}
