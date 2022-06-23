#include <d2frontend/loop_detector.h>
#include <chrono> 
#include <d2frontend/utils.h>
#include <d2common/d2vinsframe.h>

using namespace std::chrono; 

#define USE_FUNDMENTAL
#define MAX_LOOP_ID 100000000

namespace D2FrontEnd {

void LoopDetector::processImageArray(VisualImageDescArray & flatten_desc) {
    TicToc tt;
    static double t_sum = 0;
    static int t_count = 0;
    
    auto start = high_resolution_clock::now();
    
    if (t0 < 0) {
        t0 = flatten_desc.stamp;
    }

    if (flatten_desc.images.size() == 0) {
        ROS_WARN("[SWARM_LOOP] FlattenDesc must carry more than zero images");
        return;
    }

    ego_motion_traj.push(ros::Time(flatten_desc.stamp), flatten_desc.pose_drone);

    int drone_id = flatten_desc.drone_id;
    int images_num = flatten_desc.images.size();

    if (drone_id!= this->self_id && databaseSize() == 0) {
        ROS_INFO("[SWARM_LOOP] Empty local database, where giveup remote image");
        return;
    } else {
        if (loop_cam->getCameraConfiguration() == STEREO_FISHEYE) {
            ROS_INFO("[SWARM_LOOP] Detector start process KeyFrame from %d with %d images and landmark: %d", drone_id, flatten_desc.images.size(), 
                flatten_desc.spLandmarkNum());
        } else {
            ROS_INFO("[SWARM_LOOP] Detector start process KeyFrame from %d with landmark: %d and lm desc size %d", drone_id,
                flatten_desc.images[0].spLandmarkNum(), flatten_desc.images[0].landmark_descriptor.size());
        }
    }

    bool new_node = all_nodes.find(flatten_desc.drone_id) == all_nodes.end();

    all_nodes.insert(flatten_desc.drone_id);

    int dir_count = 0;
    for (auto & img : flatten_desc.images) {
        if (img.spLandmarkNum() > 0) {
            dir_count ++;
        }
    }

    if (dir_count < _config.MIN_DIRECTION_LOOP) {
        ROS_INFO("[SWARM_LOOP] Give up frame_desc with less than %d(%d) available images",
            _config.MIN_DIRECTION_LOOP, dir_count);
        return;
    }

    if (flatten_desc.spLandmarkNum() >= _config.MIN_LOOP_NUM) {
        bool init_mode = false;
        if (drone_id != self_id) {
            init_mode = true;
            if (inter_drone_loop_count[drone_id][self_id] >= _config.inter_drone_init_frames) {
                init_mode = false;
            }
        }

        //Initialize images for visualization
        if (params->show) {
            std::vector<cv::Mat> imgs;
            for (unsigned int i = 0; i < images_num; i++) {
                auto & img_des = flatten_desc.images[i];
                if (!img_des.raw_image.empty()) {
                    imgs.emplace_back(img_des.raw_image);
                } else if (img_des.image.size() != 0) {
                    imgs.emplace_back(decode_image(img_des));
                } else {
                    // imgs[i] = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
                    imgs.emplace_back(cv::Mat(params->height, params->width, CV_8U, cv::Scalar(255)));
                }
                if (params->camera_configuration == STEREO_PINHOLE) {
                    break;
                }
            }
            msgid2cvimgs[flatten_desc.frame_id] = imgs;
        }

        if (!flatten_desc.prevent_adding_db || new_node) {
            addToDatabase(flatten_desc);
        } else {
            ROS_DEBUG("[SWARM_LOOP] This image is prevent to adding to DB");
        }

        bool success = false;

        if (databaseSize() > _config.MATCH_INDEX_DIST || init_mode || drone_id != self_id) {

            ROS_INFO("[SWARM_LOOP] Querying image from database size %d init_mode %d nonkeyframe %d", databaseSize(), init_mode, flatten_desc.prevent_adding_db);
            int camera_index = 1;
            int camera_index_old = -1;
            VisualImageDescArray & _old_fisheye_img = queryDescArrayFromDatabase(flatten_desc, camera_index, camera_index_old);
            auto stop = high_resolution_clock::now(); 
            if (camera_index_old >= 0 ) {
                swarm_msgs::LoopEdge ret;
                if (_old_fisheye_img.drone_id == self_id) {
                    success = computeLoop(_old_fisheye_img, flatten_desc, camera_index_old, camera_index, ret);
                } else if (flatten_desc.drone_id == self_id) {
                    success = computeLoop(flatten_desc, _old_fisheye_img, camera_index, camera_index_old, ret);
                } else {
                    ROS_WARN("[SWARM_LOOP] Will not compute loop, drone id is %d(self %d)", flatten_desc.drone_id, self_id);
                }

                if (success) {
                    onLoopConnection(ret);
                }
            } else {
                std::cout << "[SWARM_LOOP] No matched image" << std::endl;
            }      
        } 

        // std::cout << "LOOP Detector cost" << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 <<"ms" << std::endl;
    } else {
        ROS_WARN("[SWARM_LOOP] Frame contain too less landmark %d, give up", flatten_desc.spLandmarkNum());
    }

    t_sum += tt.toc();
    t_count += 1;
    ROS_INFO("[SWARM_LOOP] Full LoopDetect avg %.1fms cur %.1fms", t_sum/t_count, tt.toc());
}


cv::Mat LoopDetector::decode_image(const VisualImageDesc & _img_desc) {
    
    auto start = high_resolution_clock::now();
    // auto ret = cv::imdecode(_img_desc.image, cv::IMREAD_GRAYSCALE);
    auto ret = cv::imdecode(_img_desc.image, cv::IMREAD_UNCHANGED);
    // std::cout << "IMDECODE Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" << std::endl;

    return ret;
}

int LoopDetector::addToDatabase(VisualImageDescArray & new_fisheye_desc) {
    for (size_t i = 0; i < new_fisheye_desc.images.size(); i++) {
        auto & img_desc = new_fisheye_desc.images[i];
        if (img_desc.spLandmarkNum() > 0 && img_desc.image_desc.size() > 0) {
            int index = addToDatabase(img_desc);
            index_to_frame_id[index] = new_fisheye_desc.frame_id;
            imgid2dir[index] = i;
            // ROS_INFO("[SWARM_LOOP] Add keyframe from %d(dir %d) to local keyframe database index: %d", img_desc.drone_id, i, index);
        }
        if (params->camera_configuration == CameraConfig::PINHOLE_DEPTH) {
            break;
        }
        keyframe_database[new_fisheye_desc.frame_id] = new_fisheye_desc;
    }
    return new_fisheye_desc.frame_id;
}

int LoopDetector::addToDatabase(VisualImageDesc & new_img_desc) {
    if (new_img_desc.drone_id == self_id) {
        local_index.add(1, new_img_desc.image_desc.data());
        return local_index.ntotal - 1;
    } else {
        remote_index.add(1, new_img_desc.image_desc.data());
        return remote_index.ntotal - 1 + REMOTE_MAGIN_NUMBER;
    }
    return -1;
}


int LoopDetector::queryFrameIndexFromDatabase(const VisualImageDesc & img_desc, double & distance) {
    double thres = _config.INNER_PRODUCT_THRES;
    int ret = -1;
    if (img_desc.drone_id == self_id) {
        //Then this is self drone
        ret = queryIndexFromDatabase(img_desc, remote_index, true, thres, _config.MATCH_INDEX_DIST, distance);
        if (ret < 0)
            ret = queryIndexFromDatabase(img_desc, local_index, false, thres, _config.MATCH_INDEX_DIST, distance);
    } else {
        ret = queryIndexFromDatabase(img_desc, local_index, false, thres, _config.MATCH_INDEX_DIST, distance);
    }
    return ret;
}

int LoopDetector::queryIndexFromDatabase(const VisualImageDesc & img_desc, faiss::IndexFlatIP & index, bool remote_db, double thres, int max_index, double & distance) {
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
    int return_frame_id = -1, return_drone_id = -1;
    int k = -1;
    for (int i = 0; i < search_num; i++) {
        if (labels[i] < 0) {
            continue;
        }

        if (index_to_frame_id.find(labels[i] + index_offset) == index_to_frame_id.end()) {
            ROS_WARN("[SWARM_LOOP] Can't find image %d; skipping", labels[i] + index_offset);
            continue;
        }

        // int return_frame_id = index_to_frame_id.at(labels[i] + index_offset);
        return_frame_id = labels[i] + index_offset;
        return_drone_id = keyframe_database[index_to_frame_id[return_frame_id]].drone_id;

        //ROS_INFO("Return Label %d/%d/%d from %d, distance %f/%f", labels[i] + index_offset, index.ntotal, index.ntotal - max_index , return_drone_id, distances[i], thres);
        
        if (labels[i] <= index.ntotal - max_index && distances[i] > thres) {
            //Is same id, max index make sense
            k = i;
            thres = distance = distances[i];
            return return_frame_id;
        }
    }

    // ROS_INFO("Database return %ld on drone %d, radius %f frame_id %d", labels[k] + index_offset, return_drone_id, distances[k], return_frame_id);
    return return_frame_id;
}


VisualImageDescArray & LoopDetector::queryDescArrayFromDatabase(const VisualImageDescArray & new_img_desc,
    int & camera_index_new, int & camera_index_old) {
    double best_distance = -1;
    int best_image_id = -1;
    //Strict use camera_index 1 now
    camera_index_new = 0;
    if (loop_cam->getCameraConfiguration() == CameraConfig::STEREO_FISHEYE) {
        camera_index_new = 1;
    } else if (
        loop_cam->getCameraConfiguration() == CameraConfig::STEREO_PINHOLE ||
        loop_cam->getCameraConfiguration() == CameraConfig::PINHOLE_DEPTH
    ) {
        camera_index_new = 0;
    } else {
        ROS_ERROR("[SWARM_LOOP] Camera configuration %d not support yet in queryDescArrayFromDatabase", loop_cam->getCameraConfiguration());
        exit(-1);
    }

    if (new_img_desc.images[camera_index_new].spLandmarkNum() > 0) {
        double distance = -1;
        int id = queryFrameIndexFromDatabase(new_img_desc.images.at(camera_index_new), distance);
        if (id != -1 && distance > best_distance) {
            best_image_id = id;
        }

        if (best_image_id != -1) {
            int frame_id = index_to_frame_id[best_image_id];
            camera_index_old = imgid2dir[best_image_id];
            ROS_INFO("[SWARM_LOOP] Query image: frame_id %d index %d drone %d with camera %d dist %f", 
                frame_id, best_image_id, keyframe_database[frame_id].drone_id, camera_index_old, distance);
            return keyframe_database[frame_id];
        }
    }

    camera_index_old = -1;
    VisualImageDescArray ret;
    ret.frame_id = -1;
    return ret;
}


int LoopDetector::databaseSize() const {
    return local_index.ntotal + remote_index.ntotal;
}


bool LoopDetector::checkLoopOdometryConsistency(LoopEdge & loop_conn) const {
    if (loop_conn.drone_id_a != loop_conn.drone_id_b || _config.DEBUG_NO_REJECT) {
        //Is inter_loop, odometry consistency check is disabled.
        return true;
    }

    Swarm::LoopEdge edge(loop_conn);
    auto odom = ego_motion_traj.get_relative_pose_by_ts(edge.ts_a, edge.ts_b);
    Eigen::Matrix6d cov_vec = odom.second + edge.get_covariance();
    auto dp = Swarm::Pose::DeltaPose(edge.relative_pose, odom.first);
    auto md = Swarm::computeSquaredMahalanobisDistance(dp.log_map(), cov_vec);
    if (md >  _config.odometry_consistency_threshold) {
        ROS_INFO("[SWARM_LOOP] LoopEdge-Odometry consistency check failed %.1f, odom %s loop %s dp %s.", 
            md, odom.first.toStr().c_str(), edge.relative_pose.toStr().c_str(), dp.toStr().c_str());
        return false;
    }

    ROS_INFO("[SWARM_LOOP] LoopEdge-Odometry consistency OK %.1f odom %s loop %s dp %s.", md, 
        odom.first.toStr().c_str(), edge.relative_pose.toStr().c_str(), dp.toStr().c_str());
    return true;
}

//Note! here the norms are both projected to main dir's unit sphere.
//index2dirindex store the dir and the index of the point
bool LoopDetector::computeCorrespondFeaturesOnImageArray(const VisualImageDescArray & frame_array_a,
    const VisualImageDescArray & frame_array_b, 
    int main_dir_a,
    int main_dir_b,
    std::vector<cv::Point3f> &lm_pos_a,
    std::vector<cv::Point2f> &lm_norm_2d_b,
    std::vector<std::pair<int, int>> &index2dirindex_a,
    std::vector<std::pair<int, int>> &index2dirindex_b
) {
    std::vector<int> dirs_a;
    std::vector<int> dirs_b;
    
    if (params->camera_configuration == STEREO_PINHOLE || params->camera_configuration == PINHOLE_DEPTH) {
        dirs_a = {0};
        dirs_b = {0};
    } else {
        for (int _dir_a = main_dir_a; _dir_a < main_dir_a + _config.MAX_DIRS; _dir_a ++) {
            int dir_a = _dir_a % _config.MAX_DIRS;
            int dir_b = ((main_dir_b - main_dir_a + _config.MAX_DIRS) % _config.MAX_DIRS + _dir_a)% _config.MAX_DIRS;
            if (dir_a < frame_array_a.images.size() && dir_b < frame_array_b.images.size()) {
                printf(" [%d: %d](%d:%d) OK", dir_b, dir_a, frame_array_b.images[dir_b].spLandmarkNum(), frame_array_a.images[dir_a].spLandmarkNum());
                if (frame_array_b.images[dir_b].spLandmarkNum() > 0 && frame_array_a.images[dir_a].spLandmarkNum() > 0) {
                    dirs_a.push_back(dir_a);
                    dirs_b.push_back(dir_b);
                }
            }
        }
    }

    printf("\n");

    Swarm::Pose extrinsic_a(frame_array_a.images[main_dir_a].extrinsic);
    Swarm::Pose extrinsic_b(frame_array_b.images[main_dir_b].extrinsic);
    Eigen::Quaterniond main_quat_new =  extrinsic_a.att();
    Eigen::Quaterniond main_quat_old =  extrinsic_b.att();


    int matched_dir_count = 0;

    for (size_t i = 0; i < dirs_a.size(); i++) {
        int dir_a = dirs_a[i];
        int dir_b = dirs_b[i];
        std::vector<cv::Point2f> lm_norm_2d_a;
        std::vector<cv::Point3f> _lm_pos_a;
        std::vector<int> _idx_a;
        std::vector<cv::Point2f> _lm_norm_2d_b;
        std::vector<int> _idx_b;

        if (dir_a < frame_array_a.images.size() && dir_b < frame_array_b.images.size()) {
            computeCorrespondFeatures(frame_array_a.images.at(dir_a),frame_array_b.images.at(dir_b),
                _lm_pos_a, _idx_a, _lm_norm_2d_b, _idx_b);
            ROS_INFO("[SWARM_LOOP] computeCorrespondFeatures on camera_index %d:%d gives %d common features", dir_b, dir_a, _lm_pos_a.size());
        } else {
            ROS_INFO("[SWARM_LOOP]  computeCorrespondFeatures on camera_index %d:%d failed: no such image");
        }

        if ( _lm_pos_a.size() >= _config.MIN_MATCH_PRE_DIR ) {
            matched_dir_count ++;            
        }

        lm_pos_a.insert(lm_pos_a.end(), _lm_pos_a.begin(), _lm_pos_a.end());
        lm_norm_2d_b.insert(lm_norm_2d_b.end(), _lm_norm_2d_b.begin(), _lm_norm_2d_b.end());

        Swarm::Pose _extrinsic_a(frame_array_a.images[dir_a].extrinsic);
        Swarm::Pose _extrinsic_b(frame_array_b.images[dir_b].extrinsic);

        if (params->camera_configuration == STEREO_FISHEYE) {
            Eigen::Quaterniond dq_new = main_quat_new.inverse() * _extrinsic_a.att();
            Eigen::Quaterniond dq_old = main_quat_old.inverse() * _extrinsic_b.att();
            for (size_t id = 0; id < lm_norm_2d_b.size(); id++) {
                auto pt = lm_norm_2d_b[id];
                // std::cout << "PT " << pt << " ROTATED " << rotate_pt_norm2d(pt, dq_old) << std::endl;
                index2dirindex_a.push_back(std::make_pair(dir_a, _idx_a[id]));
                index2dirindex_b.push_back(std::make_pair(dir_b, _idx_b[id]));
                lm_norm_2d_b[i] = rotate_pt_norm2d(pt, dq_old);
            }
        } else {
            for (size_t id = 0; id < lm_norm_2d_b.size(); id++) {
                index2dirindex_a.push_back(std::make_pair(dir_a, _idx_a[id]));
                index2dirindex_b.push_back(std::make_pair(dir_b, _idx_b[id]));
            }
        }
    }

    if(lm_norm_2d_b.size() > 0 && matched_dir_count >= _config.MIN_DIRECTION_LOOP) {
        return true;
    } else {
        return false;
    }
}

bool LoopDetector::computeCorrespondFeatures(const VisualImageDesc & new_img_desc, const VisualImageDesc & old_img_desc, 
        std::vector<cv::Point3f> &lm_pos_a,
        std::vector<int> &idx_a,
        std::vector<cv::Point2f> &lm_norm_2d_b,
        std::vector<int> &idx_b) {
    // ROS_INFO("[SWARM_LOOP](LoopDetector::computeCorrespondFeatures) %d %d ", new_img_desc.spLandmarkNum(), new_img_desc.landmark_descriptor.size());
    assert(new_img_desc.spLandmarkNum() * FEATURE_DESC_SIZE == new_img_desc.landmark_descriptor.size() && "Desciptor size of new img desc must equal to to landmarks*256!!!");
    assert(old_img_desc.spLandmarkNum() * FEATURE_DESC_SIZE == old_img_desc.landmark_descriptor.size() && "Desciptor size of old img desc must equal to to landmarks*256!!!");

    auto & _old_lms = old_img_desc.landmarks;
    auto & _now_lms = new_img_desc.landmarks;

    cv::Mat desc_now( new_img_desc.spLandmarkNum(), FEATURE_DESC_SIZE, CV_32F);
    memcpy(desc_now.data, new_img_desc.landmark_descriptor.data(), new_img_desc.landmark_descriptor.size()*sizeof(float));

    cv::Mat desc_old( old_img_desc.spLandmarkNum(), FEATURE_DESC_SIZE, CV_32F);
    memcpy(desc_old.data, old_img_desc.landmark_descriptor.data(), old_img_desc.landmark_descriptor.size()*sizeof(float));
    
    cv::BFMatcher bfmatcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> _matches;
    std::vector<unsigned char> mask;
    bfmatcher.match(desc_now, desc_old, _matches);

    std::vector<cv::Point2f> old_2d, new_2d;
    for (auto match : _matches) {
        int now_id = match.queryIdx;
        int old_id = match.trainIdx;
        if (new_img_desc.landmarks[now_id].flag == LandmarkFlag::ESTIMATED || true) {
            auto landmark_id = _now_lms[now_id].landmark_id;
            if (landmark_db.find(landmark_id) == landmark_db.end()) {
                ROS_WARN("[SWARM_LOOP] landmark_id %d not found in landmark_db", landmark_id);
                continue;
            }
            if (landmark_db.at(landmark_id).flag != LandmarkFlag::ESTIMATED) {
                // ROS_WARN("Landmark %ld is not estimated", landmark_id);
                continue;
            }
            Vector3d pt3d_norm_old = _now_lms[now_id].pt3d_norm;
            if (fabs(pt3d_norm_old(2)) > 1e-3) {
                Vector2d pt2d_norm_old = pt3d_norm_old.head(2)/pt3d_norm_old(2);
                new_2d.push_back(_now_lms[now_id].pt2d);
                old_2d.push_back(_old_lms[old_id].pt2d);

                idx_a.push_back(now_id);
                idx_b.push_back(old_id);
                lm_pos_a.push_back(toCV(landmark_db.at(landmark_id).position));
                lm_norm_2d_b.push_back(toCV(pt2d_norm_old));
            } else {
                ROS_WARN("[SWARM_LOOP] landmark_id %d is near image plane, give up in current framework", landmark_id);
            }
        }
    }

    if (old_2d.size() < 4) {
        return false;
    }
    if (_config.enable_homography_test) {
        cv::findHomography(old_2d, new_2d, cv::RANSAC, 3, mask);
        reduceVector(idx_a, mask);
        reduceVector(idx_b, mask);
        reduceVector(lm_pos_a, mask);
        reduceVector(lm_norm_2d_b, mask);
    }
    return true;
}

//Require 3d points of frame a and 2d point of frame b
bool LoopDetector::computeLoop(const VisualImageDescArray & frame_array_a, const VisualImageDescArray & frame_array_b,
    int main_dir_a, int main_dir_b, LoopEdge & ret) {

    if (frame_array_a.spLandmarkNum() < _config.MIN_LOOP_NUM) {
        return false;
    }
    //Recover imformation

    assert(frame_array_b.drone_id == self_id && "old img desc must from self drone to provide more 2d points!");

    bool success = false;

    double t_b = frame_array_b.stamp - t0;
    double t_a = frame_array_a.stamp - t0;
    ROS_INFO("[LoopDetector::computeLoop@%d] Compute loop drone b %d(d%d,dir %d)->a %d(d%d,dir %d) t %.1f->%.1f(%.1f)s landmarks %d:%d.", 
        self_id, frame_array_b.frame_id, frame_array_b.drone_id, main_dir_b, frame_array_a.frame_id, frame_array_a.drone_id, main_dir_a,
        t_b, t_a, t_a - t_b, frame_array_b.spLandmarkNum(), frame_array_a.spLandmarkNum());

    std::vector<cv::Point3f> lm_pos_a, lm_pos_b;
    std::vector<cv::Point2f> lm_norm_2d_b;
    std::vector<int> dirs_a;
    Swarm::Pose DP_old_to_new;
    std::vector<int> inliers;
    std::vector<std::pair<int, int>> index2dirindex_b;
    std::vector<std::pair<int, int>> index2dirindex_a;
    int inlier_num = 0;
    
    success = computeCorrespondFeaturesOnImageArray(frame_array_a, frame_array_b, 
        main_dir_a, main_dir_b, lm_pos_a, lm_norm_2d_b, index2dirindex_a, index2dirindex_b);
    
    if(success) {
        if (lm_pos_a.size() > _config.MIN_LOOP_NUM) {
            success = computeRelativePose( lm_pos_a, lm_norm_2d_b, frame_array_b.images[main_dir_b].extrinsic,
                    frame_array_a.pose_drone, frame_array_b.pose_drone, DP_old_to_new, inliers, _config.is_4dof);
        } else {
            ROS_INFO("[LoopDetector::computeLoop@%d]Too less common feature %ld, will give up", self_id, lm_pos_a.size());
            success = false;
        }
    } 
    else {
        ROS_INFO("[LoopDetector::computeLoop@%d] computeCorrespondFeatures failed", self_id);
        success = false;
    }

    if (params->show) {
        drawMatched(frame_array_a, frame_array_b, main_dir_a, main_dir_b, inliers, success, inlier_num, DP_old_to_new, index2dirindex_a, index2dirindex_b);
    }

    if (success) {
        ret.relative_pose = DP_old_to_new.to_ros_pose();

        ret.drone_id_a = frame_array_b.drone_id;
        ret.ts_a = ros::Time(frame_array_b.stamp);

        ret.drone_id_b = frame_array_a.drone_id;
        ret.ts_b = ros::Time(frame_array_a.stamp);

        ret.self_pose_a = toROSPose(frame_array_b.pose_drone);
        ret.self_pose_b = toROSPose(frame_array_a.pose_drone);

        ret.keyframe_id_a = frame_array_b.frame_id;
        ret.keyframe_id_b = frame_array_a.frame_id;

        ret.pos_cov.x = _config.loop_cov_pos;
        ret.pos_cov.y = _config.loop_cov_pos;
        ret.pos_cov.z = _config.loop_cov_pos;

        ret.ang_cov.x = _config.loop_cov_ang;
        ret.ang_cov.y = _config.loop_cov_ang;
        ret.ang_cov.z = _config.loop_cov_ang;

        ret.pnp_inlier_num = inlier_num;
        ret.id = self_id*MAX_LOOP_ID + loop_count;

        if (checkLoopOdometryConsistency(ret)) {
            loop_count ++;
            ROS_INFO("[SWARM_LOOP] Loop %ld Detected %d->%d dt %3.3fs DPos %4.3f %4.3f %4.3f Dyaw %3.2fdeg inliers %d. Will publish\n",
                ret.id,
                ret.drone_id_a, ret.drone_id_b,
                (ret.ts_b - ret.ts_a).toSec(),
                DP_old_to_new.pos().x(), DP_old_to_new.pos().y(), DP_old_to_new.pos().z(),
                DP_old_to_new.yaw()*57.3,
                ret.pnp_inlier_num
            );

            int new_d_id = frame_array_a.drone_id;
            int old_d_id = frame_array_b.drone_id;
            inter_drone_loop_count[new_d_id][old_d_id] = inter_drone_loop_count[new_d_id][old_d_id] +1;
            inter_drone_loop_count[old_d_id][new_d_id] = inter_drone_loop_count[old_d_id][new_d_id] +1;
            
            return true;

        } else {
            ROS_INFO("[SWARM_LOOP] Loop not consistency with odometry, give up.");
        }
    }
    return false;
}

void LoopDetector::drawMatched(const VisualImageDescArray & frame_array_a, 
            const VisualImageDescArray & frame_array_b, int main_dir_a, int main_dir_b, 
            std::vector<int> inliers, bool success, int inlier_num, Swarm::Pose DP_b_to_a,
            std::vector<std::pair<int, int>> index2dirindex_a,
            std::vector<std::pair<int, int>> index2dirindex_b) {
    cv::Mat show;
    char title[100] = {0};
    std::vector<cv::Mat> _matched_imgs;
    auto & imgs_a = msgid2cvimgs[frame_array_a.frame_id];
    auto & imgs_b = msgid2cvimgs[frame_array_b.frame_id];
    _matched_imgs.resize(imgs_b.size());
    for (size_t i = 0; i < imgs_b.size(); i ++) {
        int dir_a = ((-main_dir_b + main_dir_a + _config.MAX_DIRS) % _config.MAX_DIRS + i)% _config.MAX_DIRS;
        if (!imgs_b[i].empty() && !imgs_a[dir_a].empty()) {
            cv::vconcat(imgs_b[i], imgs_a[dir_a], _matched_imgs[i]);
        }
    } 

    // for (auto it : index2dirindex_a) {
    //     auto i = it.first;
    //     int old_pt_id = index2dirindex_b[i].second;
    //     int old_dir_id = index2dirindex_b[i].first;

    //     int new_pt_id = index2dirindex_a[i].second;
    //     int new_dir_id = index2dirindex_a[i].first;
    //     auto pt_old = frame_array_b.images[old_dir_id].landmarks[old_pt_id].pt2d;
    //     auto pt_new = frame_array_a.images[new_dir_id].landmarks[new_pt_id].pt2d;

    //     cv::line(_matched_imgs[old_dir_id], pt_old, pt_new + cv::Point2f(0, imgs_b[old_dir_id].rows), cv::Scalar(0, 0, 255));
    //     cv::circle(_matched_imgs[old_dir_id], pt_old, 3, cv::Scalar(255, 0, 0), 1);
    //     cv::circle(_matched_imgs[old_dir_id], pt_new + cv::Point2f(0, imgs_b[old_dir_id].rows), 3, cv::Scalar(255, 0, 0), 1);
    // }

    for (auto inlier: inliers) {
        int old_pt_id = index2dirindex_b[inlier].second;
        int old_dir_id = index2dirindex_b[inlier].first;

        int new_pt_id = index2dirindex_a[inlier].second;
        int new_dir_id = index2dirindex_a[inlier].first;
        auto pt_old = frame_array_b.images[old_dir_id].landmarks[old_pt_id].pt2d;
        auto pt_new = frame_array_a.images[new_dir_id].landmarks[new_pt_id].pt2d;
        // printf("inlier %d %d->%d pt %.1f %.1f -> %.1f %1.f\n", inlier, old_pt_id, new_pt_id, pt_old.x, pt_old.y, pt_new.x, pt_new.y);
        if (_matched_imgs[old_dir_id].empty()) {
            continue;
        }
        if (_matched_imgs[old_dir_id].channels() != 3) {
            cv::cvtColor(_matched_imgs[old_dir_id], _matched_imgs[old_dir_id], cv::COLOR_GRAY2BGR);
        }

        cv::line(_matched_imgs[old_dir_id], pt_old, pt_new + cv::Point2f(0, imgs_b[old_dir_id].rows), cv::Scalar(0, 255, 0));
        cv::circle(_matched_imgs[old_dir_id], pt_old, 3, cv::Scalar(255, 0, 0), 1);
        cv::circle(_matched_imgs[new_dir_id], pt_new + cv::Point2f(0, imgs_b[old_dir_id].rows), 3, cv::Scalar(255, 0, 0), 1);
    }
    

    show = _matched_imgs[0];
    for (size_t i = 1; i < _matched_imgs.size(); i ++) {
        if (_matched_imgs[i].empty()) continue;
        cv::line(_matched_imgs[i], cv::Point2f(0, 0), cv::Point2f(0, _matched_imgs[i].rows), cv::Scalar(255, 255, 0), 2);
        cv::hconcat(show, _matched_imgs[i], show);
    }

    double dt = (frame_array_a.stamp - frame_array_b.stamp);
    if (success) {
        auto ypr = DP_b_to_a.rpy()*180/M_PI;
        sprintf(title, "MAP-BASED EDGE %d->%d dt %3.3fs inliers %d", 
            frame_array_b.drone_id, frame_array_a.drone_id, dt, inlier_num);
        cv::putText(show, title, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);

        sprintf(title, "T %.2f %.2f %.2f YPR %.1f %.1f %.1f", 
            DP_b_to_a.pos().x(), DP_b_to_a.pos().y(), DP_b_to_a.pos().z(),
            ypr.z(), ypr.y(), ypr.x());
        cv::putText(show, title, cv::Point2f(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
        sprintf(title, "%d<->%d", 
            frame_array_b.frame_id,
            frame_array_a.frame_id);
        cv::putText(show, title, cv::Point2f(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
        
        } else {
        sprintf(title, "FAILED LOOP %d->%d dt %3.3fs inliers %d", frame_array_b.drone_id, frame_array_a.drone_id, dt, inlier_num);
        cv::putText(show, title, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.5);
    }

    // cv::resize(show, show, cv::Size(), 2, 2);
    static int loop_match_count = 0;
    loop_match_count ++;
    char PATH[100] = {0};

    if (!show.empty() && success) {
        sprintf(PATH, "loop/match%d.png", loop_match_count);
        cv::imwrite(params->OUTPUT_PATH+PATH, show);
    }
    cv::imshow("Matches", show);
    cv::waitKey(10);
}
void LoopDetector::onLoopConnection(LoopEdge & loop_conn) {
    on_loop_cb(loop_conn);
}

void LoopDetector::updatebyLandmarkDB(const std::map<LandmarkIdType, LandmarkPerId> & vins_landmark_db) {
    for (auto it : vins_landmark_db) {
        auto landmark_id = it.first;
        if (landmark_db.find(landmark_id) == landmark_db.end()) {
            landmark_db[landmark_id] = it.second;
        } else {
            if (it.second.flag == LandmarkFlag::ESTIMATED) {
                landmark_db[landmark_id] = it.second;
            }
        }
    }
}

void LoopDetector::updatebySldWin(const std::vector<VINSFrame*> sld_win) {
    for (auto frame : sld_win) {
        auto frame_id = frame->frame_id;
        if (keyframe_database.find(frame_id) != keyframe_database.end()) {
            keyframe_database.at(frame_id).pose_drone = frame->odom.pose();
        }
    }
}

LoopDetector::LoopDetector(int _self_id, const LoopDetectorConfig & config):
        self_id(_self_id),
        _config(config),
        local_index(DEEP_DESC_SIZE), 
        remote_index(DEEP_DESC_SIZE), 
    ego_motion_traj(_self_id, true, _config.pos_covariance_per_meter, _config.yaw_covariance_per_meter) {
}

bool pnp_result_verify(bool pnp_success, int inliers, double rperr, const Swarm::Pose & DP_old_to_new) {
    bool success = pnp_success;
    if (!pnp_success) {
        return false;
    }
    if (rperr > RPERR_THRES) {
        ROS_INFO("[SWARM_LOOP] Check failed on RP error %f", rperr*57.3);
        return false;
    }   
    auto &_config = (*params->loopdetectorconfig);
    success = (inliers >= _config.MIN_LOOP_NUM) && fabs(DP_old_to_new.yaw()) < ACCEPT_LOOP_YAW_RAD && DP_old_to_new.pos().norm() < MAX_LOOP_DIS;
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
    return RPerr;
}

int computeRelativePose(const std::vector<cv::Point3f> lm_positions_a, const std::vector<cv::Point2f> lm_2d_norm_b,
        Swarm::Pose extrinsic_b, Swarm::Pose drone_pose_a, Swarm::Pose drone_pose_b, Swarm::Pose & DP_b_to_a, std::vector<int> &inliers, bool is_4dof) {
        //Compute PNP
    // ROS_INFO("Matched features %ld", matched_2d_norm_old.size());
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    cv::Mat r, rvec, rvec2, t, t2, D, tmp_r;

    int iteratives = 100;

    bool success = solvePnPRansac(lm_positions_a, lm_2d_norm_b, K, D, rvec, t, false,   
        iteratives,  3, 0.99,  inliers);
    auto p_cam_old_in_new = PnPRestoCamPose(rvec, t);
    auto p_drone_old_in_new = p_cam_old_in_new*(extrinsic_b.to_isometry().inverse());
    if (!success) {
        return 0;
    }
    DP_b_to_a =  Swarm::Pose::DeltaPose(p_drone_old_in_new, drone_pose_a, is_4dof);
    auto RPerr = RPerror(p_drone_old_in_new, drone_pose_b, drone_pose_a);
    success = pnp_result_verify(success, inliers.size(), RPerr, DP_b_to_a);
    ROS_INFO("[SWARM_LOOP] DPose %s PnPRansac %d inlines %d/%d, dyaw %f dpos %f. Geometry Check %f",
        DP_b_to_a.toStr().c_str(), success, inliers.size(), lm_2d_norm_b.size(), 
        fabs(DP_b_to_a.yaw())*57.3, DP_b_to_a.pos().norm(), RPerr);
    return success;
}


}