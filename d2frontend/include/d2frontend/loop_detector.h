#pragma once

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <swarm_msgs/LoopEdge.h>
#include <d2frontend/d2frontend_params.h>
#include <d2frontend/loop_cam.h>
#include <functional>
#include <swarm_msgs/Pose.h>
#include <faiss/IndexFlat.h>
#include <swarm_msgs/drone_trajectory.hpp>

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
    double netvlad_IP_thres;
    double DETECTOR_MATCH_THRES;
    int inter_drone_init_frames;
    bool DEBUG_NO_REJECT;
    double odometry_consistency_threshold;
    int loop_inlier_feature_num;
    bool is_4dof;
    double pos_covariance_per_meter;
    double yaw_covariance_per_meter;
    bool enable_homography_test = false;
    bool enable_superglue = true;
    double accept_loop_max_yaw = 15;
    double accept_loop_max_pos = 1.5;
    std::string superglue_model_path;
};

class SuperGlueOnnx;

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
    SuperGlueOnnx * superglue = nullptr;
    
    //Use 3D points from frame a.
    bool computeLoop(const VisualImageDescArray & frame_array_a, const VisualImageDescArray & frame_array_b,
        int main_dir_a, int main_dir_b, LoopEdge & ret);

    bool computeCorrespondFeatures(const VisualImageDesc & new_img_desc, const VisualImageDesc & old_img_desc, 
        Point3fVector &lm_pos_a, std::vector<int> &idx_a, Point2fVector &lm_norm_2d_b, std::vector<int> &idx_b);

    bool computeCorrespondFeaturesOnImageArray(const VisualImageDescArray & frame_array_a,
        const VisualImageDescArray & frame_array_b, int main_dir_a, int main_dir_b,
        Point3fVector &lm_pos_a, Point2fVector &lm_norm_2d_b, std::vector<std::pair<int, int>> &index2dirindex_a,
        std::vector<std::pair<int, int>> &index2dirindex_b);

    int addToDatabase(VisualImageDescArray & new_fisheye_desc);
    int addToDatabase(VisualImageDesc & new_img_desc);
    bool queryDescArrayFromDatabase(const VisualImageDescArray & new_img_desc, VisualImageDescArray & ret, int & camera_index_new, int & camera_index_old);
    int queryFrameIndexFromDatabase(const VisualImageDesc & new_img_desc, double & similarity);
    int queryIndexFromDatabase(const VisualImageDesc & new_img_desc, faiss::IndexFlatIP & index, bool remote_db, double thres, int max_index, double & similarity);


    std::set<int> all_nodes;

    bool checkLoopOdometryConsistency(LoopEdge & loop_conn) const;
    Swarm::DroneTrajectory ego_motion_traj;

    void drawMatched(const VisualImageDescArray & fisheye_desc_a, const VisualImageDescArray & fisheye_desc_b,
            int main_dir_a, int main_dir_b, bool success, std::vector<int> inliers, Swarm::Pose DP_b_to_a,
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
    
int computeRelativePose(const Point3fVector lm_positions_a, const Point2fVector lm_2d_norm_b,
        Swarm::Pose extrinsic_b, Swarm::Pose drone_pose_a, Swarm::Pose drone_pose_b, Swarm::Pose & DP_b_to_a,
        std::vector<int> &inliers, bool is_4dof);

}