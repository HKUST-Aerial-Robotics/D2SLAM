#include <swarm_loop/loop_detector.h>
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include <opencv2/opencv.hpp>
#include <chrono> 

using namespace std::chrono; 

#define USE_FUNDMENTAL
#define MAX_LOOP_ID 100000000

void LoopDetector::on_image_recv(const FisheyeFrameDescriptor_t & flatten_desc, std::vector<cv::Mat> imgs) {
    TicToc tt;
    static double t_sum = 0;
    static int t_count = 0;
    
    auto start = high_resolution_clock::now();
    
    if (t0 < 0) {
        t0 = toROSTime(flatten_desc.timestamp).toSec();
    }

    if (flatten_desc.images.size() == 0) {
        ROS_WARN("[SWARM_LOOP] FlattenDesc must carry more than zero images");
        return;
    }

    ego_motion_traj.push(toROSTime(flatten_desc.timestamp), Swarm::Pose(flatten_desc.pose_drone));

    int drone_id = flatten_desc.drone_id;
    int images_num = flatten_desc.images.size();

    if (imgs.size() < images_num) {
        imgs.resize(images_num);
    }
    
    if (drone_id!= this->self_id && database_size() == 0) {
        ROS_INFO("[SWARM_LOOP] Empty local database, where giveup remote image");
        return;
    } else {
        if (loop_cam->get_camera_configuration() == STEREO_FISHEYE) {
            ROS_INFO("[SWARM_LOOP] Detector start process KeyFrame from %d with %d images and landmark: %d", drone_id, flatten_desc.images.size(), 
                flatten_desc.landmark_num);
        } else {
            ROS_INFO("[SWARM_LOOP] Detector start process KeyFrame from %d with landmark: %d and lm desc size %d", drone_id,
                flatten_desc.images[0].landmarks_2d.size(), flatten_desc.images[0].feature_descriptor.size());
        }
    }

    bool new_node = all_nodes.find(flatten_desc.drone_id) == all_nodes.end();

    all_nodes.insert(flatten_desc.drone_id);

    int dir_count = 0;
    for (auto & img : flatten_desc.images) {
        if (img.landmark_num > 0) {
            dir_count ++;
        }
    }

    if (dir_count < MIN_DIRECTION_LOOP) {
        ROS_INFO("[SWARM_LOOP] Give up frame_desc with less than %d(%d) available images", MIN_DIRECTION_LOOP, dir_count);
        return;
    }

    if (flatten_desc.landmark_num >= MIN_LOOP_NUM) {
        bool init_mode = false;
        if (drone_id != self_id) {
            init_mode = true;
            if (inter_drone_loop_count[drone_id][self_id] >= inter_drone_init_frames) {
                init_mode = false;
            }
        }

        //Initialize images for visualization
        if (enable_visualize) {
            for (unsigned int i = 0; i < images_num; i++) {
                auto & img_des = flatten_desc.images[i];
                if (imgs[i].empty()) {
                    if (img_des.image.size() != 0) {
                        imgs[i] = decode_image(img_des);
                    } else {
                        // imgs[i] = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
                        imgs[i] = cv::Mat(height, width, CV_8U, cv::Scalar(255));
                    }
                }
            }
        }

        if (!flatten_desc.prevent_adding_db || new_node) {
            add_to_database(flatten_desc);
            msgid2cvimgs[flatten_desc.msg_id] = imgs;
        } else {
            ROS_DEBUG("[SWARM_LOOP] This image is prevent to adding to DB");
        }

        bool success = false;

        if (database_size() > MATCH_INDEX_DIST || init_mode || drone_id != self_id) {

            ROS_INFO("[SWARM_LOOP] Querying image from database size %d init_mode %d nonkeyframe %d", database_size(), init_mode, flatten_desc.prevent_adding_db);
            
            int direction = 1;
            int direction_old = -1;
            FisheyeFrameDescriptor_t & _old_fisheye_img = query_fisheyeframe_from_database(flatten_desc, init_mode, flatten_desc.prevent_adding_db, direction, direction_old);
            auto stop = high_resolution_clock::now(); 

            if (direction_old >= 0 ) {
                swarm_msgs::LoopEdge ret;

                if (_old_fisheye_img.drone_id == self_id) {
                    success = compute_loop(flatten_desc, _old_fisheye_img, direction, direction_old, imgs, msgid2cvimgs[_old_fisheye_img.msg_id], ret, init_mode);
                } else {
                    //We grab remote drone from database
                    if (flatten_desc.drone_id == self_id) {
                        success = compute_loop(_old_fisheye_img, flatten_desc, direction_old, direction, msgid2cvimgs[_old_fisheye_img.msg_id],  imgs, ret, init_mode);
                    } else {
                        ROS_WARN("[SWARM_LOOP] Will not compute loop, drone id is %d(self %d)", flatten_desc.drone_id, self_id);
                    }
                }

                if (success) {
                    on_loop_connection(ret);
                }
            } else {
                std::cout << "[SWARM_LOOP] No matched image" << std::endl;
            }      
        } 

        // std::cout << "LOOP Detector cost" << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 <<"ms" << std::endl;
    } else {
        ROS_WARN("[SWARM_LOOP] Frame contain too less landmark %d, give up", flatten_desc.landmark_num);
    }

    t_sum += tt.toc();
    t_count += 1;
    ROS_INFO("[SWARM_LOOP] Full LoopDetect avg %.1fms cur %.1fms", t_sum/t_count, tt.toc());
}


cv::Mat LoopDetector::decode_image(const ImageDescriptor_t & _img_desc) {
    
    auto start = high_resolution_clock::now();
    // auto ret = cv::imdecode(_img_desc.image, cv::IMREAD_GRAYSCALE);
    auto ret = cv::imdecode(_img_desc.image, cv::IMREAD_UNCHANGED);
    // std::cout << "IMDECODE Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" << std::endl;

    return ret;
}

int LoopDetector::add_to_database(const FisheyeFrameDescriptor_t & new_fisheye_desc) {
    for (size_t i = 0; i < new_fisheye_desc.images.size(); i++) {
        auto & img_desc = new_fisheye_desc.images[i];
        if (img_desc.landmark_num > 0) {
            int index = add_to_database(img_desc);
            imgid2fisheye[index] = new_fisheye_desc.msg_id;
            imgid2dir[index] = i;
            // ROS_INFO("[SWARM_LOOP] Add keyframe from %d(dir %d) to local keyframe database index: %d", img_desc.drone_id, i, index);
        }
    }
    fisheyeframe_database[new_fisheye_desc.msg_id] = new_fisheye_desc;
    return new_fisheye_desc.msg_id;
}

int LoopDetector::add_to_database(const ImageDescriptor_t & new_img_desc) {
    if (new_img_desc.drone_id == self_id) {
        local_index.add(1, new_img_desc.image_desc.data());
        return local_index.ntotal - 1;
    } else {
        remote_index.add(1, new_img_desc.image_desc.data());
        return remote_index.ntotal - 1 + REMOTE_MAGIN_NUMBER;
    }
    return -1;
}


int LoopDetector::query_from_database(const ImageDescriptor_t & img_desc, bool init_mode, bool nonkeyframe, double & distance) {
    double thres = INNER_PRODUCT_THRES;
    if (init_mode) {
        thres = INIT_MODE_PRODUCT_THRES;
    }

    if (img_desc.drone_id == self_id) {
        //Then this is self drone
        int _id = query_from_database(img_desc, remote_index, true, thres, 1, distance);
        if(!nonkeyframe){
            int _id = query_from_database(img_desc, local_index, false, thres, MATCH_INDEX_DIST, distance);
            return _id;
        } else if (_id != -1) {
            return _id;
        } 
    } else {
        int _id = query_from_database(img_desc, local_index, false, thres, 1, distance);
        // ROS_INFO("Is remote image, query only from remote db: %d", _id);
        return _id;
    }
    return -1;
}

int LoopDetector::query_from_database(const ImageDescriptor_t & img_desc, faiss::IndexFlatIP & index, bool remote_db, double thres, int max_index, double & distance) {
    float distances[1000] = {0};
    faiss::Index::idx_t labels[1000];

    int index_offset = 0;
    if (remote_db) {
        index_offset = REMOTE_MAGIN_NUMBER;
    }
    
    for (int i = 0; i < 1000; i++) {
        labels[i] = -1;
    }

    int search_num = SEARCH_NEAREST_NUM + max_index;
    index.search(1, img_desc.image_desc.data(), search_num, distances, labels);
    int return_msg_id = -1, return_drone_id = -1;
    int k = -1;
    for (int i = 0; i < search_num; i++) {
        if (labels[i] < 0) {
            continue;
        }

        if (imgid2fisheye.find(labels[i] + index_offset) == imgid2fisheye.end()) {
            ROS_WARN("[SWARM_LOOP] Can't find image %d; skipping", labels[i] + index_offset);
            continue;
        }

        // int return_msg_id = imgid2fisheye.at(labels[i] + index_offset);
        return_msg_id = labels[i] + index_offset;
        return_drone_id = fisheyeframe_database[imgid2fisheye[return_msg_id]].drone_id;

        //ROS_INFO("Return Label %d/%d/%d from %d, distance %f/%f", labels[i] + index_offset, index.ntotal, index.ntotal - max_index , return_drone_id, distances[i], thres);
        
        if (labels[i] <= index.ntotal - max_index && distances[i] > thres) {
            //Is same id, max index make sense
            k = i;
            thres = distance = distances[i];
            return return_msg_id;
        }
    }

    // ROS_INFO("Database return %ld on drone %d, radius %f msg_id %d", labels[k] + index_offset, return_drone_id, distances[k], return_msg_id);
    return return_msg_id;
}


FisheyeFrameDescriptor_t & LoopDetector::query_fisheyeframe_from_database(const FisheyeFrameDescriptor_t & new_img_desc, bool init_mode, bool nonkeyframe, int & direction_new, int & direction_old) {
    double best_distance = -1;
    int best_image_id = -1;
    //Strict use direction 1 now
    direction_new = 0;
    if (loop_cam->get_camera_configuration() == CameraConfig::STEREO_FISHEYE) {
        direction_new = 1;
    } else if (
        loop_cam->get_camera_configuration() == CameraConfig::STEREO_PINHOLE ||
        loop_cam->get_camera_configuration() == CameraConfig::PINHOLE_DEPTH
    ) {
        direction_new = 0;
    } else {
        ROS_ERROR("[SWARM_LOOP] Camera configuration %d not support yet in query_fisheyeframe_from_database", loop_cam->get_camera_configuration());
        exit(-1);
    }

    if (new_img_desc.images[direction_new].landmark_num > 0) {
        double distance = -1;
        int id = query_from_database(new_img_desc.images.at(direction_new), init_mode, nonkeyframe, distance);
        if (id != -1 && distance > best_distance) {
            best_image_id = id;
        }

        // ROS_INFO("query_from_database(new_img_desc.images.at(direction_new) return %d best_image_id %d distance %f/%f", 
            // id, best_image_id, distance, best_distance);


        if (best_image_id != -1) {
            int msg_id = imgid2fisheye[best_image_id];
            direction_old = imgid2dir[best_image_id];
            FisheyeFrameDescriptor_t & ret = fisheyeframe_database[msg_id];
            ROS_INFO("[SWARM_LOOP] Database return image %d fisheye frame from drone %d with direction %d dist %f", 
                best_image_id, ret.drone_id, direction_old, distance);
            return ret;
        }
    }

    direction_old = -1;
    FisheyeFrameDescriptor_t ret;
    ret.msg_id = -1;
    return ret;
}


int LoopDetector::database_size() const {
    return local_index.ntotal + remote_index.ntotal;
}


bool LoopDetector::check_loop_odometry_consistency(LoopEdge & loop_conn) const {
    if (loop_conn.drone_id_a != loop_conn.drone_id_b || DEBUG_NO_REJECT) {
        //Is inter_loop, odometry consistency check is disabled.
        return true;
    }

    Swarm::LoopEdge edge(loop_conn);
    auto odom = ego_motion_traj.get_relative_pose_by_ts(edge.ts_a, edge.ts_b);
    Eigen::Matrix6d cov_vec = odom.second + edge.get_covariance();
    auto dp = Swarm::Pose::DeltaPose(edge.relative_pose, odom.first);
    auto md = Swarm::computeSquaredMahalanobisDistance(dp.log_map(), cov_vec);
    if (md > odometry_consistency_threshold) {
        ROS_INFO("[SWARM_LOOP] LoopEdge-Odometry consistency check failed %.1f, odom %s loop %s dp %s.", 
            md, odom.first.tostr().c_str(), edge.relative_pose.tostr().c_str(), dp.tostr().c_str());
        return false;
    }

    ROS_INFO("[SWARM_LOOP] LoopEdge-Odometry consistency OK %.1f odom %s loop %s dp %s.", md, 
        odom.first.tostr().c_str(), edge.relative_pose.tostr().c_str(), dp.tostr().c_str());
    return true;
}

bool pnp_result_verify(bool pnp_success, bool init_mode, int inliers, double rperr, const Swarm::Pose & DP_old_to_new) {
    bool success = pnp_success;
    if (!pnp_success) {
        return false;
    }

    if (rperr > RPERR_THRES) {
        ROS_INFO("[SWARM_LOOP] Check failed on RP error %f", rperr*57.3);
        return false;
    }

    if (init_mode) {
        success = (inliers >= INIT_MODE_MIN_LOOP_NUM) && fabs(DP_old_to_new.yaw()) < ACCEPT_LOOP_YAW_RAD && DP_old_to_new.pos().norm() < MAX_LOOP_DIS;            
    } else {
        success = (inliers >= MIN_LOOP_NUM) && fabs(DP_old_to_new.yaw()) < ACCEPT_LOOP_YAW_RAD && DP_old_to_new.pos().norm() < MAX_LOOP_DIS;
    }        

    return success;
}


double RPerror(const Swarm::Pose & p_drone_old_in_new, const Swarm::Pose & drone_pose_old, const Swarm::Pose & drone_pose_now) {
    Swarm::Pose DP_old_to_new_6d =  Swarm::Pose::DeltaPose(p_drone_old_in_new, drone_pose_now, false);
    Swarm::Pose Prediect_new_in_old_Pose = drone_pose_old * DP_old_to_new_6d;
    auto AttNew_in_old = Prediect_new_in_old_Pose.att().normalized();
    auto AttNew_in_new = drone_pose_now.att().normalized();
    auto dyaw = quat2eulers(AttNew_in_new).z() - quat2eulers(AttNew_in_old).z();
    AttNew_in_old = Eigen::AngleAxisd(dyaw, Eigen::Vector3d::UnitZ())*AttNew_in_old;
    auto RPerr = (quat2eulers(AttNew_in_old) - quat2eulers(AttNew_in_new)).norm();
    // std::cout << "New In Old" << quat2eulers(AttNew_in_old) << std::endl;
    // std::cout << "New In New"  << quat2eulers(AttNew_in_new);
    // std::cout << "Estimate RP error" <<  (quat2eulers(AttNew_in_old) - quat2eulers(AttNew_in_new))*57.3 << std::endl;
    // std::cout << "Estimate RP error2" <<  quat2eulers( (AttNew_in_old.inverse()*AttNew_in_new).normalized())*57.3 << std::endl;
    return RPerr;
}

 

int LoopDetector::compute_relative_pose(
        const std::vector<cv::Point2f> matched_2d_norm_now,
        const std::vector<cv::Point3f> matched_3d_now,
        const std::vector<cv::Point2f> matched_2d_norm_old,
        const std::vector<cv::Point3f> matched_3d_old,
        Swarm::Pose old_extrinsic,
        Swarm::Pose drone_pose_now,
        Swarm::Pose drone_pose_old,
        Swarm::Pose & DP_old_to_new,
        bool init_mode,
        int drone_id_new, int drone_id_old,
        std::vector<cv::DMatch> &matches,
        int &inlier_num) {
        //Compute PNP
    // ROS_INFO("Matched features %ld", matched_2d_norm_old.size());
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);

    cv::Mat r, rvec, rvec2, t, t2, D, tmp_r;
    cv::Mat inliers;


    Swarm::Pose initial_old_drone_pose = drone_pose_old;
    Swarm::Pose initial_old_cam_pose = initial_old_drone_pose * old_extrinsic;
    // Swarm::Pose old_cam_in_new_initial = drone_pose_now.inverse() * initial_old_cam_pose;
    // Swarm::Pose old_drone_to_new_initial = drone_pose_old.inverse() * drone_pose_now;
    // std::cout << "OLD to new initial" << std::endl;
    // old_drone_to_new_initial.print();
    // PnPInitialFromCamPose(initial_old_cam_pose, rvec, t);
    
    int iteratives = 100;

    if (init_mode) {
        iteratives = 1000;
    }

    bool success = solvePnPRansac(matched_3d_now, matched_2d_norm_old, K, D, rvec, t, false,   
        iteratives,  3, 0.99,  inliers);

    auto p_cam_old_in_new = PnPRestoCamPose(rvec, t);
    auto p_drone_old_in_new = p_cam_old_in_new*(old_extrinsic.to_isometry().inverse());
    
    if (!success) {
        return 0;
    }

    DP_old_to_new =  Swarm::Pose::DeltaPose(p_drone_old_in_new, drone_pose_now, is_4dof);
    
    auto RPerr = RPerror(p_drone_old_in_new, drone_pose_old, drone_pose_now);

    success = pnp_result_verify(success, init_mode, inliers.rows, RPerr, DP_old_to_new);

    ROS_INFO("[SWARM_LOOP] DPose %s PnPRansac %d inlines %d/%d, dyaw %f dpos %f. Geometry Check %f", DP_old_to_new.tostr().c_str(), success, inliers.rows, matched_2d_norm_old.size(), fabs(DP_old_to_new.yaw())*57.3, DP_old_to_new.pos().norm(), RPerr);
    inlier_num = inliers.rows;

    for (int i = 0; i < inlier_num; i++) {
        matches.push_back(cv::DMatch(inliers.at<int>(i, 0), inliers.at<int>(i, 0), 0));
    }
    return success;
}

cv::Point2f rotate_pt_norm2d(cv::Point2f pt, Eigen::Quaterniond q) {
    Eigen::Vector3d pt3d(pt.x, pt.y, 1);
    pt3d = q * pt3d;

    if (pt3d.z() < 1e-3 && pt3d.z() > 0) {
        pt3d.z() = 1e-3;
    }

    if (pt3d.z() > -1e-3 && pt3d.z() < 0) {
        pt3d.z() = -1e-3;
    }

    return cv::Point2f(pt3d.x()/ pt3d.z(), pt3d.y()/pt3d.z());
}

//Note! here the norms are both projected to main dir's unit sphere.
bool LoopDetector::compute_correspond_features(const FisheyeFrameDescriptor_t & new_frame_desc,
    const FisheyeFrameDescriptor_t & old_frame_desc, 
    int main_dir_new,
    int main_dir_old,
    std::vector<cv::Point2f> &new_norm_2d,
    std::vector<cv::Point3f> &new_3d,
    std::vector<std::vector<int>> &new_idx,
    std::vector<cv::Point2f> &old_norm_2d,
    std::vector<cv::Point3f> &old_3d,
    std::vector<std::vector<int>> &old_idx,
    std::vector<int> &dirs_new,
    std::vector<int> &dirs_old,
    std::map<int, std::pair<int, int>> &index2dirindex_new,
    std::map<int, std::pair<int, int>> &index2dirindex_old
) {
    //For each FisheyeFrameDescriptor_t, there must be 4 frames
    //However, due to the transmission and parameter, some may be empty.
    // We will only matched the frame which isn't empty

    printf("compute_correspond_features on main dir [%d(drone%d): %d(drone%d)]: ",
        main_dir_old, old_frame_desc.drone_id,
        main_dir_new, new_frame_desc.drone_id
    );

    for (int _dir_new = main_dir_new; _dir_new < main_dir_new + MAX_DIRS; _dir_new ++) {
        int dir_new = _dir_new % MAX_DIRS;
        int dir_old = ((main_dir_old - main_dir_new + MAX_DIRS) % MAX_DIRS + _dir_new)% MAX_DIRS;
        if (dir_new < new_frame_desc.images.size() && dir_old < old_frame_desc.images.size()) {
            printf(" [%d: %d](%d:%d) OK", dir_old, dir_new, old_frame_desc.images[dir_old].landmark_num, new_frame_desc.images[dir_new].landmark_num );
            if (old_frame_desc.images[dir_old].landmark_num > 0 && new_frame_desc.images[dir_new].landmark_num > 0) {
                dirs_new.push_back(dir_new);
                dirs_old.push_back(dir_old);
            }
        }
    }

    printf("\n");

    Swarm::Pose extrinsic_new(new_frame_desc.images[main_dir_new].camera_extrinsic);
    Swarm::Pose extrinsic_old(old_frame_desc.images[main_dir_old].camera_extrinsic);
    Eigen::Quaterniond main_quat_new =  extrinsic_new.att();
    Eigen::Quaterniond main_quat_old =  extrinsic_old.att();


    int matched_dir_count = 0;

    for (size_t i = 0; i < dirs_new.size(); i++) {
        int dir_new = dirs_new[i];
        int dir_old = dirs_old[i];
        std::vector<cv::Point2f> _new_norm_2d;
        std::vector<cv::Point3f> _new_3d;
        std::vector<int> _new_idx;
        std::vector<cv::Point2f> _old_norm_2d;
        std::vector<cv::Point3f> _old_3d;
        std::vector<int> _old_idx;

        if (dir_new < new_frame_desc.images.size() && dir_old < old_frame_desc.images.size()) {
            compute_correspond_features(
                new_frame_desc.images.at(dir_new),
                old_frame_desc.images.at(dir_old),
                _new_norm_2d,
                _new_3d,
                _new_idx,
                _old_norm_2d,
                _old_3d,
                _old_idx
            );
            ROS_INFO("[SWARM_LOOP] compute_correspond_features on direction %d:%d gives %d common features", dir_old, dir_new, _new_3d.size());
        } else {
            ROS_INFO("[SWARM_LOOP]  compute_correspond_features on direction %d:%d failed: no such image");
        }

        if ( _new_3d.size() >= MIN_MATCH_PRE_DIR ) {
            matched_dir_count ++;            
        }

        new_3d.insert(new_3d.end(), _new_3d.begin(), _new_3d.end());
        old_3d.insert(old_3d.end(), _old_3d.begin(), _old_3d.end());
        new_idx.push_back(_new_idx);
        old_idx.push_back(_old_idx);

        Swarm::Pose _extrinsic_new(new_frame_desc.images[dir_new].camera_extrinsic);
        Swarm::Pose _extrinsic_old(old_frame_desc.images[dir_old].camera_extrinsic);

        Eigen::Quaterniond dq_new = main_quat_new.inverse() * _extrinsic_new.att();
        Eigen::Quaterniond dq_old = main_quat_old.inverse() * _extrinsic_old.att();

        for (size_t id = 0; id < _old_norm_2d.size(); id++) {
            auto pt = _old_norm_2d[id];
            // std::cout << "PT " << pt << " ROTATED " << rotate_pt_norm2d(pt, dq_old) << std::endl;
            index2dirindex_old[old_norm_2d.size()] = std::make_pair(dir_old, _old_idx[id]);
            old_norm_2d.push_back(rotate_pt_norm2d(pt, dq_old));
        }

        for (size_t id = 0; id < _new_norm_2d.size(); id++) {
            auto pt = _new_norm_2d[id];
            index2dirindex_new[new_norm_2d.size()] = std::make_pair(dir_new, _new_idx[id]);
            new_norm_2d.push_back(rotate_pt_norm2d(pt, dq_new));
        }
    }

    if(new_norm_2d.size() > 0 && matched_dir_count >= MIN_DIRECTION_LOOP) {
        return true;
    } else {
        return false;
    }
}

bool LoopDetector::compute_correspond_features(const ImageDescriptor_t & new_img_desc, const ImageDescriptor_t & old_img_desc, 
        std::vector<cv::Point2f> &new_norm_2d,
        std::vector<cv::Point3f> &new_3d,
        std::vector<int> &new_idx,
        std::vector<cv::Point2f> &old_norm_2d,
        std::vector<cv::Point3f> &old_3d,
        std::vector<int> &old_idx) {
    // ROS_INFO("[SWARM_LOOP](LoopDetector::compute_correspond_features) %d %d ", new_img_desc.landmarks_2d.size(), new_img_desc.feature_descriptor.size());
    assert(new_img_desc.landmarks_2d.size() * FEATURE_DESC_SIZE == new_img_desc.feature_descriptor.size() && "Desciptor size of new img desc must equal to to landmarks*256!!!");
    assert(old_img_desc.landmarks_2d.size() * FEATURE_DESC_SIZE == old_img_desc.feature_descriptor.size() && "Desciptor size of old img desc must equal to to landmarks*256!!!");

    auto _old_norm_2d = toCV(old_img_desc.landmarks_2d_norm);
    auto _old_2d = toCV(old_img_desc.landmarks_2d);
    auto _old_3d = toCV(old_img_desc.landmarks_3d);
    
    auto _now_norm_2d = toCV(new_img_desc.landmarks_2d_norm);
    auto _now_2d = toCV(new_img_desc.landmarks_2d);
    auto _now_3d = toCV(new_img_desc.landmarks_3d);

    cv::Mat desc_now( _now_norm_2d.size(), FEATURE_DESC_SIZE, CV_32F);
    memcpy(desc_now.data, new_img_desc.feature_descriptor.data(), new_img_desc.feature_descriptor.size()*sizeof(float));

    cv::Mat desc_old( old_img_desc.landmarks_2d.size(), FEATURE_DESC_SIZE, CV_32F);
    memcpy(desc_old.data, old_img_desc.feature_descriptor.data(), old_img_desc.feature_descriptor.size()*sizeof(float));
    
    cv::BFMatcher bfmatcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> _matches;
    std::vector<unsigned char> mask;
    bfmatcher.match(desc_now, desc_old, _matches);

#ifdef USE_FUNDMENTAL
    std::vector<cv::Point2f> old_2d, new_2d;
    for (auto match : _matches) {
        int now_id = match.queryIdx;
        int old_id = match.trainIdx;
        if (new_img_desc.landmarks_flag[now_id]) {
            new_2d.push_back(_now_2d[now_id]);
            old_2d.push_back(_old_2d[old_id]);

            new_idx.push_back(now_id);
            old_idx.push_back(old_id);

            new_3d.push_back(_now_3d[now_id]);
            new_norm_2d.push_back(_now_norm_2d[now_id]);

            old_3d.push_back(_old_3d[old_id]);
            old_norm_2d.push_back(_old_norm_2d[old_id]);
        }
    }

    if (old_2d.size() >= 4) {
        cv::findHomography(old_2d, new_2d, CV_RANSAC, 3, mask);
        reduceVector(new_idx, mask);
        reduceVector(old_idx, mask);

        reduceVector(new_3d, mask);
        reduceVector(new_norm_2d, mask);

        reduceVector(old_3d, mask);
        reduceVector(old_norm_2d, mask);
    } else {
        return false;
    }

    return true;
#else
    for (auto match : _matches) {
        int now_id = match.queryIdx;
        int old_id = match.trainIdx;
        if (match.distance < DETECTOR_MATCH_THRES && new_img_desc.landmarks_flag[now_id]) {

            new_idx.push_back(now_id);
            old_idx.push_back(old_id);

            new_3d.push_back(_now_3d[now_id]);
            new_norm_2d.push_back(_now_norm_2d[now_id]);

            old_3d.push_back(_old_3d[old_id]);
            old_norm_2d.push_back(_old_norm_2d[old_id]);
        } else {
            // printf("Give up distance too high %f\n", match.distance);
        }
    }
    return true;
#endif
}

//Require 3d points of new frame and 2d point of old frame
bool LoopDetector::compute_loop(const FisheyeFrameDescriptor_t & new_frame_desc, const FisheyeFrameDescriptor_t & old_frame_desc,
    int main_dir_new, int main_dir_old,
    std::vector<cv::Mat> imgs_new, std::vector<cv::Mat> imgs_old,
    LoopEdge & ret, bool init_mode) {

    if (new_frame_desc.landmark_num < MIN_LOOP_NUM) {
        return false;
    }
    //Recover imformation

    assert(old_frame_desc.drone_id == self_id && "old img desc must from self drone to provide more 2d points!");

    bool success = false;

    double told = toROSTime(old_frame_desc.timestamp).toSec() - t0;
    double tnew = toROSTime(new_frame_desc.timestamp).toSec() - t0;
    ROS_INFO("Compute loop drone %d(dir %d)->%d(dir %d) t %f->%f(%f) msgid %d->%d landmarks %d:%d. Init %d", old_frame_desc.drone_id, main_dir_old, new_frame_desc.drone_id, main_dir_new,
        told, tnew, tnew - told,
        old_frame_desc.msg_id, new_frame_desc.msg_id,
        old_frame_desc.landmark_num,
        new_frame_desc.landmark_num,
        init_mode);

    std::vector<cv::Point2f> new_norm_2d;
    std::vector<cv::Point3f> new_3d;
    std::vector<std::vector<int>> new_idx;
    std::vector<cv::Point2f> old_norm_2d;
    std::vector<cv::Point3f> old_3d;
    std::vector<std::vector<int>> old_idx;
    std::vector<int> dirs_new;
    std::vector<int> dirs_old;
    Swarm::Pose DP_old_to_new;
    std::vector<cv::DMatch> matches;
    std::map<int, std::pair<int, int>> index2dirindex_old;
    std::map<int, std::pair<int, int>> index2dirindex_new;
    int inlier_num = 0;
    
    success = compute_correspond_features(new_frame_desc, old_frame_desc, 
        main_dir_new, main_dir_old,
        new_norm_2d, new_3d, new_idx,
        old_norm_2d, old_3d, old_idx, dirs_new, dirs_old, 
        index2dirindex_new, index2dirindex_old);
    
    if(success) {
        if (new_norm_2d.size() > MIN_LOOP_NUM || (init_mode && new_norm_2d.size() > INIT_MODE_MIN_LOOP_NUM)) {
            success = compute_relative_pose(
                    new_norm_2d, new_3d, 
                    old_norm_2d, old_3d,
                    Swarm::Pose(old_frame_desc.images[main_dir_old].camera_extrinsic),
                    Swarm::Pose(new_frame_desc.pose_drone),
                    Swarm::Pose(old_frame_desc.pose_drone),
                    DP_old_to_new,
                    init_mode,
                    new_frame_desc.drone_id,
                    old_frame_desc.drone_id,
                    matches,
                    inlier_num
            );
        } else {
            ROS_INFO("Too less common feature %ld, will give up", new_norm_2d.size());
            success = false;
        }
    } 
    else {
        ROS_INFO("compute_correspond_features failed");
        success = false;
    }

    if (enable_visualize) {
        cv::Mat show;
        char title[100] = {0};
        std::vector<cv::Mat> _matched_imgs;
        _matched_imgs.resize(imgs_old.size());
        for (size_t i = 0; i < imgs_old.size(); i ++) {
            int dir_new = ((-main_dir_old + main_dir_new + MAX_DIRS) % MAX_DIRS + i)% MAX_DIRS;
            if (!imgs_old[i].empty() && !imgs_new[dir_new].empty()) {
                cv::vconcat(imgs_old[i], imgs_new[dir_new], _matched_imgs[i]);
            }
        } 

        for (size_t i = 0; i < new_norm_2d.size(); i ++) {
            int old_pt_id = index2dirindex_old[i].second;
            int old_dir_id = index2dirindex_old[i].first;

            int new_pt_id = index2dirindex_new[i].second;
            int new_dir_id = index2dirindex_new[i].first;
            auto pt_old = toCV(old_frame_desc.images[old_dir_id].landmarks_2d[old_pt_id]);
            auto pt_new = toCV(new_frame_desc.images[new_dir_id].landmarks_2d[new_pt_id]);

            cv::line(_matched_imgs[old_dir_id], pt_old, pt_new + cv::Point2f(0, imgs_old[old_dir_id].rows), cv::Scalar(0, 0, 255));
            cv::circle(_matched_imgs[old_dir_id], pt_old, 3, cv::Scalar(255, 0, 0), 1);
            cv::circle(_matched_imgs[old_dir_id], pt_new + cv::Point2f(0, imgs_old[old_dir_id].rows), 3, cv::Scalar(255, 0, 0), 1);
        
        }

        for (auto match: matches) {
            int idi = match.queryIdx;
            int idj = match.trainIdx;
            int old_pt_id = index2dirindex_old[idi].second;
            int old_dir_id = index2dirindex_old[idi].first;

            int new_pt_id = index2dirindex_new[idi].second;
            int new_dir_id = index2dirindex_new[idi].first;
            auto pt_old = toCV(old_frame_desc.images[old_dir_id].landmarks_2d[old_pt_id]);
            auto pt_new = toCV(new_frame_desc.images[new_dir_id].landmarks_2d[new_pt_id]);
            if (_matched_imgs[old_dir_id].empty()) {
                continue;
            }
            if (_matched_imgs[old_dir_id].channels() != 3) {
                cv::cvtColor(_matched_imgs[old_dir_id], _matched_imgs[old_dir_id], cv::COLOR_GRAY2BGR);
            }

            cv::line(_matched_imgs[old_dir_id], pt_old, pt_new + cv::Point2f(0, imgs_old[old_dir_id].rows), cv::Scalar(0, 255, 0));
            cv::circle(_matched_imgs[old_dir_id], pt_old, 3, cv::Scalar(255, 0, 0), 1);
            cv::circle(_matched_imgs[new_dir_id], pt_new + cv::Point2f(0, imgs_old[old_dir_id].rows), 3, cv::Scalar(255, 0, 0), 1);
        }
        

        show = _matched_imgs[0];
        for (size_t i = 1; i < _matched_imgs.size(); i ++) {
            if (_matched_imgs[i].empty()) continue;
            cv::line(_matched_imgs[i], cv::Point2f(0, 0), cv::Point2f(0, _matched_imgs[i].rows), cv::Scalar(255, 255, 0), 2);
            cv::hconcat(show, _matched_imgs[i], show);
        }

        double dt = (toROSTime(new_frame_desc.timestamp) - toROSTime(old_frame_desc.timestamp)).toSec();
        if (success) {
            auto ypr = DP_old_to_new.rpy()*180/M_PI;
            sprintf(title, "MAP-BASED EDGE %d->%d dt %3.3fs inliers %d", 
                old_frame_desc.drone_id, new_frame_desc.drone_id, dt, inlier_num);
            cv::putText(show, title, cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);

            sprintf(title, "T %.2f %.2f %.2f YPR %.1f %.1f %.1f", 
                DP_old_to_new.pos().x(), DP_old_to_new.pos().y(), DP_old_to_new.pos().z(),
                ypr.z(), ypr.y(), ypr.x());
            cv::putText(show, title, cv::Point2f(20, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
            sprintf(title, "%d<->%d", 
                old_frame_desc.msg_id,
                new_frame_desc.msg_id);
            cv::putText(show, title, cv::Point2f(20, 70), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
            
           } else {
            sprintf(title, "FAILED LOOP %d->%d dt %3.3fs inliers %d", old_frame_desc.drone_id, new_frame_desc.drone_id, dt, inlier_num);
            cv::putText(show, title, cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
        }

        // cv::resize(show, show, cv::Size(), 2, 2);
        static int loop_match_count = 0;
        loop_match_count ++;
        char PATH[100] = {0};

        if (!show.empty() && success) {
            sprintf(PATH, "loop/match%d.png", loop_match_count);
            cv::imwrite(OUTPUT_PATH+PATH, show);
            
            cv::imshow("Matches", show);
            cv::waitKey(10);
        }
    }

    if (success) {
        ret.relative_pose = DP_old_to_new.to_ros_pose();

        ret.drone_id_a = old_frame_desc.drone_id;
        ret.ts_a = toROSTime(old_frame_desc.timestamp);

        ret.drone_id_b = new_frame_desc.drone_id;
        ret.ts_b = toROSTime(new_frame_desc.timestamp);

        ret.self_pose_a = toROSPose(old_frame_desc.pose_drone);
        ret.self_pose_b = toROSPose(new_frame_desc.pose_drone);

        ret.keyframe_id_a = old_frame_desc.msg_id;
        ret.keyframe_id_b = new_frame_desc.msg_id;

        ret.pos_cov.x = loop_cov_pos;
        ret.pos_cov.y = loop_cov_pos;
        ret.pos_cov.z = loop_cov_pos;

        ret.ang_cov.x = loop_cov_ang;
        ret.ang_cov.y = loop_cov_ang;
        ret.ang_cov.z = loop_cov_ang;

        ret.pnp_inlier_num = inlier_num;
        ret.id = self_id*MAX_LOOP_ID + loop_count;

        if (check_loop_odometry_consistency(ret)) {
            loop_count ++;
            ROS_INFO("[SWARM_LOOP] Loop %ld Detected %d->%d dt %3.3fs DPos %4.3f %4.3f %4.3f Dyaw %3.2fdeg inliers %d. Will publish\n",
                ret.id,
                ret.drone_id_a, ret.drone_id_b,
                (ret.ts_b - ret.ts_a).toSec(),
                DP_old_to_new.pos().x(), DP_old_to_new.pos().y(), DP_old_to_new.pos().z(),
                DP_old_to_new.yaw()*57.3,
                ret.pnp_inlier_num
            );

            int new_d_id = new_frame_desc.drone_id;
            int old_d_id = old_frame_desc.drone_id;
            inter_drone_loop_count[new_d_id][old_d_id] = inter_drone_loop_count[new_d_id][old_d_id] +1;
            inter_drone_loop_count[old_d_id][new_d_id] = inter_drone_loop_count[old_d_id][new_d_id] +1;
            
            return true;

        } else {
            ROS_INFO("[SWARM_LOOP] Loop not consistency with odometry, give up.");
        }
    }
    return false;
}

void LoopDetector::on_loop_connection(LoopEdge & loop_conn) {
    on_loop_cb(loop_conn);
}

LoopDetector::LoopDetector(int _self_id): self_id(_self_id), local_index(DEEP_DESC_SIZE), remote_index(DEEP_DESC_SIZE), 
    ego_motion_traj(_self_id, true, pos_covariance_per_meter, yaw_covariance_per_meter) {
}
