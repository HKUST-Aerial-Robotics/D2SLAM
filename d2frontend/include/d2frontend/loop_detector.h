#pragma once

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <swarm_msgs/LoopEdge.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/loop_cam.h>
#include <functional>
#include <swarm_msgs/Pose.h>
#include <swarm_msgs/swarm_types.hpp>
#ifdef USE_DEEPNET
#include <faiss/IndexFlat.h>
#else
#include <DBoW3/DBoW3.h>
#endif

using namespace swarm_msgs;

#define REMOTE_MAGIN_NUMBER 1000000

namespace D2Frontend {

struct LoopDetectorConfig {
    int MATCH_INDEX_DIST;
    int MAX_DIRS;
    int MIN_DIRECTION_LOOP;
    int MIN_MATCH_PRE_DIR;
    double loop_cov_pos;
    double loop_cov_ang;
    double INNER_PRODUCT_THRES;
    double INIT_MODE_PRODUCT_THRES;//INIT mode we can accept this inner product as similar
    double DETECTOR_MATCH_THRES;
    int inter_drone_init_frames;
    bool DEBUG_NO_REJECT;
    double odometry_consistency_threshold;
    int INIT_MODE_MIN_LOOP_NUM; //Init mode we accepte this inlier number
    int MIN_LOOP_NUM;
    bool is_4dof;
    double pos_covariance_per_meter;
    double yaw_covariance_per_meter;
};

class LoopDetector {
    LoopDetectorConfig _config;

protected:
    faiss::IndexFlatIP local_index;

    faiss::IndexFlatIP remote_index;

    std::map<int, int64_t> imgid2fisheye;
    std::map<int, int> imgid2dir;
    std::map<int, std::map<int, int>> inter_drone_loop_count;

    std::map<int64_t, VisualImageDescArray> keyframe_database;

    std::map<int64_t, std::vector<cv::Mat>> msgid2cvimgs;
    
    double t0 = -1;
    int loop_count = 0;
    
    bool computeLoop(const VisualImageDescArray & new_fisheye_desc, const VisualImageDescArray & old_fisheye_desc,
        int main_dir_new, int main_dir_old,
        std::vector<cv::Mat> img_new, std::vector<cv::Mat> img_old, LoopEdge & ret, bool init_mode=false);

    bool computeCorrespondFeatures(const VisualImageDesc & new_img_desc, const VisualImageDesc & old_img_desc, 
        std::vector<cv::Point2f> &new_norm_2d,
        std::vector<cv::Point3f> &new_3d,
        std::vector<int> &new_idx,
        std::vector<cv::Point2f> &old_norm_2d,
        std::vector<cv::Point3f> &old_3d,
        std::vector<int> &old_idx
    );

    bool computeCorrespondFeatures(const VisualImageDescArray & new_img_desc, const VisualImageDescArray & old_img_desc, 
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

    int computeRelativePose(
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

    int addToDatabase(const VisualImageDescArray & new_fisheye_desc);
    int addToDatabase(const VisualImageDesc & new_img_desc);
    VisualImageDescArray & queryDescArrayFromFatabase(const VisualImageDescArray & new_img_desc, bool init_mode, bool nonkeyframe, int & direction_new, int & direction_old);
    int queryFromDatabase(const VisualImageDesc & new_img_desc, bool init_mode, bool nonkeyframe, double & distance);
    int queryFromDatabase(const VisualImageDesc & new_img_desc, faiss::IndexFlatIP & index, bool remote_db, double thres, int max_index, double & distance);


    std::set<int> all_nodes;

    bool checkLoopOdometryConsistency(LoopEdge & loop_conn) const;
    Swarm::DroneTrajectory ego_motion_traj;

public:
    std::function<void(LoopEdge &)> on_loop_cb;
    int self_id = -1;
    LoopDetector(int self_id, const LoopDetectorConfig & config);
    void onImageRecv(const VisualImageDescArray & img_des, std::vector<cv::Mat> img = std::vector<cv::Mat>(0));
    void onLoopConnection(LoopEdge & loop_conn);
    LoopCam * loop_cam = nullptr;
    bool enable_visualize = true;
    cv::Mat decode_image(const VisualImageDesc & _img_desc);

    int databaseSize() const;

};
}