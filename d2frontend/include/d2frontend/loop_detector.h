#pragma once

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <swarm_msgs/LoopEdge.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/loop_cam.h>
#include <functional>
#include <swarm_msgs/Pose.h>
#include <swarm_msgs/swarm_types.hpp>
#include <faiss/IndexFlat.h>

using namespace swarm_msgs;
#define REMOTE_MAGIN_NUMBER 1000000

namespace D2FrontEnd {

struct LoopDetectorConfig {
    int MATCH_INDEX_DIST;
    int MAX_DIRS;
    int MIN_DIRECTION_LOOP;
    int MIN_MATCH_PRE_DIR;
    double loop_cov_pos;
    double loop_cov_ang;
    double INNER_PRODUCT_THRES;
    double DETECTOR_MATCH_THRES;
    int inter_drone_init_frames;
    bool DEBUG_NO_REJECT;
    double odometry_consistency_threshold;
    int MIN_LOOP_NUM;
    bool is_4dof;
    double pos_covariance_per_meter;
    double yaw_covariance_per_meter;
    bool enable_homography_test = false;
};

class LoopDetector {
    LoopDetectorConfig _config;
    std::map<LandmarkIdType, LandmarkPerId> landmark_db;
protected:
    faiss::IndexFlatIP local_index;
    faiss::IndexFlatIP remote_index;

    std::map<int, int64_t> index_to_frame_id;
    std::map<int, int> imgid2dir;
    std::map<int, std::map<int, int>> inter_drone_loop_count;

    std::map<int64_t, VisualImageDescArray> keyframe_database;

    std::map<int64_t, std::vector<cv::Mat>> msgid2cvimgs;
    
    double t0 = -1;
    int loop_count = 0;
    
    //Use 3D points from frame a.
    bool computeLoop(const VisualImageDescArray & fisheye_desc_a, const VisualImageDescArray & fisheye_desc_b,
        int main_dir_a, int main_dir_b, LoopEdge & ret);

    bool computeCorrespondFeatures(const VisualImageDesc & new_img_desc, const VisualImageDesc & old_img_desc, 
        std::vector<cv::Point3f> &new_3d,
        std::vector<int> &new_idx,
        std::vector<cv::Point2f> &old_norm_2d,
        std::vector<int> &old_idx
    );

    bool computeCorrespondFeaturesOnImageArray(const VisualImageDescArray & new_img_desc, const VisualImageDescArray & old_img_desc, 
        int main_dir_a, int main_dir_b,
        std::vector<cv::Point3f> &lm_pos_a,
        std::vector<cv::Point2f> &lm_norm_2d_b,
        std::vector<std::pair<int, int>> &dirindex_new,
        std::vector<std::pair<int, int>> &dirindex_old
    );

    int addToDatabase(VisualImageDescArray & new_fisheye_desc);
    int addToDatabase(VisualImageDesc & new_img_desc);
    VisualImageDescArray & queryDescArrayFromDatabase(const VisualImageDescArray & new_img_desc, int & camera_index_new, int & camera_index_old);
    int queryFrameIndexFromDatabase(const VisualImageDesc & new_img_desc, double & distance);
    int queryIndexFromDatabase(const VisualImageDesc & new_img_desc, faiss::IndexFlatIP & index, bool remote_db, double thres, int max_index, double & distance);


    std::set<int> all_nodes;

    bool checkLoopOdometryConsistency(LoopEdge & loop_conn) const;
    Swarm::DroneTrajectory ego_motion_traj;

    void drawMatched(const VisualImageDescArray & fisheye_desc_a, const VisualImageDescArray & fisheye_desc_b,
            int main_dir_a, int main_dir_b, std::vector<int> inliers, bool success, int inlier_num, Swarm::Pose DP_b_to_a,
            std::vector<std::pair<int, int>> index2dirindex_a, std::vector<std::pair<int, int>> index2dirindex_b);

public:
    std::function<void(LoopEdge &)> on_loop_cb;
    int self_id = -1;
    LoopDetector(int self_id, const LoopDetectorConfig & config);
    void processImageArray(VisualImageDescArray & img_des);
    void onLoopConnection(LoopEdge & loop_conn);
    LoopCam * loop_cam = nullptr;
    cv::Mat decode_image(const VisualImageDesc & _img_desc);
    void updatebyLandmarkDB(const std::map<LandmarkIdType, LandmarkPerId> & vins_landmark_db);
    void updatebySldWin(const std::vector<VINSFrame*> sld_win);

    int databaseSize() const;

};
    
int computeRelativePose(const std::vector<cv::Point3f> lm_positions_a, const std::vector<cv::Point2f> lm_2d_norm_b,
        Swarm::Pose extrinsic_b, Swarm::Pose drone_pose_a, Swarm::Pose drone_pose_b, Swarm::Pose & DP_b_to_a,
        std::vector<int> &inliers, bool is_4dof);

}