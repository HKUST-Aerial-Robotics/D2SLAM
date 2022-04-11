#pragma once

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <swarm_msgs/ImageDescriptor.h>
#include <swarm_msgs/LoopEdge.h>
#include <swarm_msgs/ImageDescriptor_t.hpp>
#include "swarm_loop/loop_defines.h"
#include <swarm_loop/loop_cam.h>
#include <functional>
#include <swarm_msgs/Pose.h>
#include <swarm_msgs/FisheyeFrameDescriptor_t.hpp>
#include <swarm_msgs/swarm_types.hpp>
#ifdef USE_DEEPNET
#include <faiss/IndexFlat.h>
#else
#include <DBoW3/DBoW3.h>
#endif

using namespace swarm_msgs;

#define REMOTE_MAGIN_NUMBER 1000000

class LoopDetector {

protected:
    faiss::IndexFlatIP local_index;

    faiss::IndexFlatIP remote_index;

    std::map<int, int64_t> imgid2fisheye;
    std::map<int, int> imgid2dir;
    std::map<int, std::map<int, int>> inter_drone_loop_count;

    std::map<int64_t, FisheyeFrameDescriptor_t> fisheyeframe_database;

    std::map<int64_t, std::vector<cv::Mat>> msgid2cvimgs;
    
    double t0 = -1;
    int loop_count = 0;
    
    bool compute_loop(const FisheyeFrameDescriptor_t & new_fisheye_desc, const FisheyeFrameDescriptor_t & old_fisheye_desc,
        int main_dir_new, int main_dir_old,
        std::vector<cv::Mat> img_new, std::vector<cv::Mat> img_old, LoopEdge & ret, bool init_mode=false);

    bool compute_correspond_features(const ImageDescriptor_t & new_img_desc, const ImageDescriptor_t & old_img_desc, 
        std::vector<cv::Point2f> &new_norm_2d,
        std::vector<cv::Point3f> &new_3d,
        std::vector<int> &new_idx,
        std::vector<cv::Point2f> &old_norm_2d,
        std::vector<cv::Point3f> &old_3d,
        std::vector<int> &old_idx
    );

    bool compute_correspond_features(const FisheyeFrameDescriptor_t & new_img_desc, const FisheyeFrameDescriptor_t & old_img_desc, 
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
    );

    int compute_relative_pose(
        const std::vector<cv::Point2f> now_norm_2d,
        const std::vector<cv::Point3f> now_3d,

        const std::vector<cv::Point2f> old_norm_2d,
        const std::vector<cv::Point3f> old_3d,

        Swarm::Pose old_extrinsic,
        Swarm::Pose drone_pose_now,
        Swarm::Pose drone_pose_old,
        Swarm::Pose & DP_old_to_new,
        bool init_mode,
        int drone_id_new, int drone_id_old,
        std::vector<cv::DMatch> &matches,
        int &inlier_num
        );

    int add_to_database(const FisheyeFrameDescriptor_t & new_fisheye_desc);
    int add_to_database(const ImageDescriptor_t & new_img_desc);
    FisheyeFrameDescriptor_t & query_fisheyeframe_from_database(const FisheyeFrameDescriptor_t & new_img_desc, bool init_mode, bool nonkeyframe, int & direction_new, int & direction_old);
    int query_from_database(const ImageDescriptor_t & new_img_desc, bool init_mode, bool nonkeyframe, double & distance);
    int query_from_database(const ImageDescriptor_t & new_img_desc, faiss::IndexFlatIP & index, bool remote_db, double thres, int max_index, double & distance);


    std::set<int> all_nodes;

    bool check_loop_odometry_consistency(LoopEdge & loop_conn) const;
    Swarm::DroneTrajectory ego_motion_traj;

public:
    std::function<void(LoopEdge &)> on_loop_cb;
    int self_id = -1;
    LoopDetector(int self_id);
    void on_image_recv(const FisheyeFrameDescriptor_t & img_des, std::vector<cv::Mat> img = std::vector<cv::Mat>(0));
    void on_loop_connection(LoopEdge & loop_conn);
    LoopCam * loop_cam = nullptr;
    bool enable_visualize = true;
    cv::Mat decode_image(const ImageDescriptor_t & _img_desc);

    int database_size() const;

};
