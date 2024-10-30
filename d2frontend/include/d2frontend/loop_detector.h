#pragma once

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <swarm_msgs/LoopEdge.h>
#include <d2frontend/d2frontend_params.h>
#include <functional>
#include <swarm_msgs/Pose.h>
#include <faiss/IndexFlat.h>
#include <swarm_msgs/drone_trajectory.hpp>
#include <mutex>

using namespace swarm_msgs;
#define REMOTE_MAGIN_NUMBER 1000000

namespace D2Common {
    class VisualImageDescArray;
    class VisualImageDesc;
    struct LandmarkPerId;
    struct VINSFrame;
}

namespace D2FrontEnd {

using D2Common::FrameIdType;
using D2Common::LandmarkIdType;
using D2Common::LandmarkPerId;
using D2Common::VisualImageDesc;
using D2Common::VisualImageDescArray;
using D2Common::VINSFrame;
using D2Common::Point2fVector;

class LoopCam;

struct LoopDetectorConfig {
    int match_index_dist;
    int match_index_dist_remote;
    int MAX_DIRS;
    int MIN_DIRECTION_LOOP;
    int MIN_MATCH_PRE_DIR;
    double loop_cov_pos;
    double loop_cov_ang;
    double loop_detection_netvlad_thres;
    double DETECTOR_MATCH_THRES;
    int inter_drone_init_frames;
    bool DEBUG_NO_REJECT;
    double odometry_consistency_threshold;
    int loop_inlier_feature_num;
    bool is_4dof;
    double pos_covariance_per_meter;
    double yaw_covariance_per_meter;
    bool enable_homography_test = false;
    bool enable_superglue = false;
    double accept_loop_max_yaw = 15;
    double accept_loop_max_pos = 1.5;
    bool enable_knn_match = true;
    double knn_match_ratio = 0.8;
    double gravity_check_thres = 0.06;
    std::string superglue_model_path;
};

class SuperGlueOnnx;

class LoopDetector {
    LoopDetectorConfig _config;
    std::map<LandmarkIdType, LandmarkPerId> landmark_db;
    std::recursive_mutex frame_mutex, landmark_mutex;
protected:
    faiss::IndexFlatIP local_index;
    faiss::IndexFlatIP remote_index;
    Swarm::DroneTrajectory ego_motion_traj;
    std::map<int, int64_t> index_to_frame_id;
    std::map<int, int> imgid2dir;
    std::map<int, std::map<int, int>> inter_drone_loop_count;
    std::set<int> all_nodes;

    std::map<int64_t, VisualImageDescArray> keyframe_database;
    std::mutex keyframe_database_mutex;

    std::map<int64_t, std::vector<cv::Mat>> msgid2cvimgs;
    
    double t0 = -1;
    int loop_count = 0;
    SuperGlueOnnx * superglue = nullptr;
    
    //Use 3D points from frame a.
    bool computeLoop(const VisualImageDescArray & frame_array_a, const VisualImageDescArray & frame_array_b,
            int main_dir_a, int main_dir_b, LoopEdge & ret);

    bool computeCorrespondFeatures(const VisualImageDesc & new_img_desc, const VisualImageDesc & old_img_desc, 
            std::vector<Vector3d> &lm_pos_a, std::vector<int> &idx_a, std::vector<Vector3d> &lm_norm_3d_b, std::vector<int> &idx_b, 
            std::vector<int> &cam_indices);

    bool computeCorrespondFeaturesOnImageArray(const VisualImageDescArray & frame_array_a,
            const VisualImageDescArray & frame_array_b, int main_dir_a, int main_dir_b,
            std::vector<Vector3d> &lm_pos_a, std::vector<Vector3d> &lm_norm_3d_b, std::vector<int> & cam_indices,
            std::vector<std::pair<int, int>> &index2dirindex_a, std::vector<std::pair<int, int>> &index2dirindex_b);

    int addImageArrayToDatabase(VisualImageDescArray & new_fisheye_desc, bool add_to_faiss = true);
    int addImageDescToDatabase(VisualImageDesc & new_img_desc);
    bool queryImageArrayFromDatabase(const VisualImageDescArray & new_img_desc, VisualImageDescArray & ret, int & camera_index_new, int & camera_index_old);
    int queryFrameIndexFromDatabase(const VisualImageDesc & new_img_desc, double & similarity);
    int queryIndexFromDatabase(const VisualImageDesc & new_img_desc, faiss::IndexFlatIP & index, bool remote_db, double thres, int max_index, double & similarity);

    bool checkLoopOdometryConsistency(LoopEdge & loop_conn) const;
    void drawMatched(const VisualImageDescArray & fisheye_desc_a, const VisualImageDescArray & fisheye_desc_b,
            int main_dir_a, int main_dir_b, bool success, std::vector<int> inliers, Swarm::Pose DP_b_to_a,
            std::vector<std::pair<int, int>> index2dirindex_a, std::vector<std::pair<int, int>> index2dirindex_b);
public:
    std::function<void(LoopEdge &)> on_loop_cb;
    std::function<void(VisualImageDescArray&)> broadcast_keyframe_cb;
    int self_id = -1;
    LoopDetector(int self_id, const LoopDetectorConfig & config);
    void processImageArray(VisualImageDescArray & img_des);
    void onLoopConnection(LoopEdge & loop_conn);
    LoopCam * loop_cam = nullptr;
    cv::Mat decode_image(const VisualImageDesc & _img_desc);
    void updatebyLandmarkDB(const std::map<LandmarkIdType, LandmarkPerId> & vins_landmark_db);
    void updatebySldWin(const std::vector<VINSFrame*> sld_win);
    bool hasFrame(FrameIdType frame_id);

    int databaseSize() const;

};


}