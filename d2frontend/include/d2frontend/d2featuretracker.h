#pragma once

#include "d2frontend_params.h"
#include "d2landmark_manager.h"
#include <unordered_map>
#include <mutex>
#include <d2common/d2frontend_types.h>
#include <d2frontend/utils.h>

using namespace Eigen;

#ifdef USE_CUDA
#define LKImageInfo LKImageInfoGPU
#else
#define LKImageInfo LKImageInfoCPU
#endif

namespace D2FrontEnd {
using D2Common::VisualImageDescArray;
using D2Common::VisualImageDesc;
using D2Common::LandmarkIdType;
using D2Common::FrameIdType;

struct D2FTConfig {
    bool show_feature_id = true;
    int long_track_thres = 20;
    int long_track_frames = 4;
    int last_track_thres = 20;
    double new_feature_thres = 0.5;
    double parallex_thres = 10.0/460.0;
    int min_keyframe_num = 2;
    bool write_to_file = false;
    bool check_essential = false;
    bool enable_lk_optical_flow = true;
    bool lk_use_fast = true;
    double ransacReprojThreshold = 10;
    double max_pts_velocity_time=0.3;
    int remote_min_match_num = 30;
    int min_stereo_points = 10;
    bool double_counting_common_feature = false;
    bool enable_superglue_local = false;
    bool enable_superglue_remote = false;
    bool enable_knn_match = true;
    bool enable_search_local_aera = true;
    bool enable_motion_prediction_local = false;
    bool enable_search_local_aera_remote = false; //Enable motion prediction searching for remote drones.
    double search_local_max_dist = 0.04; //To multiply with width
    double search_local_max_dist_lr = 0.2; //To multiply with width
    double knn_match_ratio = 0.8;
    std::string output_folder = "/root/output/";
    std::string superglue_model_path;
    double landmark_distance_assumption = 100.0; // For uninitialized landmark, assume it is 100 meter away
    int frame_step = 2;
    bool track_from_keyframe = true;
    bool lr_match_use_lk = true;
    bool lk_lk_use_pred = true;
    bool sp_track_use_lk = true;

    //frontend thread frequency.
    float  stereo_frame_thread_rate = 20.0;
    float loop_detection_thread_rate = 1.0;
    float lcm_thread_rate = 1.0;
};

struct TrackReport {
    double sum_parallex = 0.0;
    int parallex_num = 0;
    int long_track_num = 0.0;
    int unmatched_num = 0.0;
    double ft_time = 0.0;
    int stereo_point_num = 0;
    int remote_matched_num = 0;

    void compose(const TrackReport & report) {
        sum_parallex += report.sum_parallex;
        parallex_num += report.parallex_num;
        long_track_num += report.long_track_num;
        unmatched_num += report.unmatched_num;
        remote_matched_num += report.remote_matched_num;
    }

    double meanParallex() const {
        return sum_parallex/parallex_num;
    }
};


class SuperGlueOnnx;

class D2FeatureTracker {
protected:
    struct MatchLocalFeatureParams {
        bool enable_prediction = false;
        bool enable_search_in_local = false;
        Swarm::Pose pose_a = Swarm::Pose();
        Swarm::Pose pose_b_prediction = Swarm::Pose();
        bool enable_superglue=true;
        TrackLRType type=WHOLE_IMG_MATCH;
        bool plot=false;
        double search_radius = 0.0;
        bool prediction_using_extrinsic = false;
    };

    D2FTConfig _config;
    double image_width = 0.0;
    double search_radius = 0.0;
    int reference_frame_id = 0;

    std::vector<VisualImageDescArray> current_keyframes;
    LandmarkManager * lmanager = nullptr;
    int keyframe_count = 0;
    int frame_count = 0;
    bool inited = false;
    std::map<int, std::map<int, LKImageInfo>> keyframe_lk_infos; //frame.camera_index->image
    std::pair<bool, LandmarkPerFrame> createLKLandmark(const VisualImageDesc & frame,
        cv::Point2f pt, LandmarkIdType landmark_id = -1, LandmarkType type=LandmarkType::FlowLandmark);
    std::recursive_mutex track_lock;
    std::recursive_mutex keyframe_lock;
    std::recursive_mutex lmanager_lock;
    
    std::map<int, std::vector<cv::Point2f>> landmark_predictions_viz;
    std::map<int, std::vector<cv::Point2f>> landmark_predictions_matched_viz;

    TrackReport trackLK(VisualImageDesc & frame);
    TrackReport track(const VisualImageDesc & left_frame, VisualImageDesc & right_frame, 
        bool enable_lk=true, TrackLRType type=WHOLE_IMG_MATCH, bool use_lk_for_left_right_track = true);
    TrackReport trackLK(const VisualImageDesc & frame, VisualImageDesc & right_frame, TrackLRType type=WHOLE_IMG_MATCH, bool use_lk_for_left_right_track = true);
    TrackReport track(VisualImageDesc & frame, const Swarm::Pose & motion_prediction=Swarm::Pose());
    TrackReport trackRemote(VisualImageDesc & frame, const VisualImageDesc & prev_frame, 
            bool use_motion_predict=false, const Swarm::Pose & motion_prediction=Swarm::Pose());
    bool getMatchedPrevKeyframe(const VisualImageDescArray & frame_a, VisualImageDescArray& prev, int & dir_a, int & dir_b);
    void processFrame(VisualImageDescArray & frames, bool is_keyframe);
    bool isKeyframe(const TrackReport & reports);
    Vector3d extractPointVelocity(const LandmarkPerFrame & lpf) const;
    std::pair<bool, LandmarkPerFrame> getPreviousLandmarkFrame(const LandmarkPerFrame & lpf, FrameIdType keyframe_id=-1) const;
    const VisualImageDescArray& getLatestKeyframe() const;

    void draw(const VisualImageDesc & frame, bool is_keyframe, const TrackReport & report) const;
    void draw(const VisualImageDesc & lframe, VisualImageDesc & rframe, bool is_keyframe, const TrackReport & report) const;
    void draw(const VisualImageDescArray & frames, bool is_keyframe, const TrackReport & report) const;
    void drawRemote(const VisualImageDescArray & frames, const TrackReport & report) const;
    void cvtRemoteLandmarkId(VisualImageDesc & frame) const;
    cv::Mat drawToImage(const VisualImageDesc & frame, bool is_keyframe, const TrackReport & report, bool is_right=false, bool is_remote=false) const;
    std::unordered_map<LandmarkIdType, LandmarkIdType> remote_to_local; // Remote landmark id to local;
    std::unordered_map<LandmarkIdType, std::unordered_map<int, LandmarkIdType>> local_to_remote; // local landmark id to remote drone and id;
    typedef std::lock_guard<std::recursive_mutex> Guard;
    SuperGlueOnnx * superglue = nullptr;
    bool matchLocalFeatures(const VisualImageDesc & img_desc_a, const VisualImageDesc & img_desc_b, std::vector<int> & ids_down_to_up, 
        const MatchLocalFeatureParams & param);
    std::map<LandmarkIdType, cv::Point2f> predictLandmarksWithExtrinsic(int camera_index, 
            std::vector<LandmarkIdType> pts_ids, std::vector<Eigen::Vector3d> pts_3d_norm, const Swarm::Pose & cam_pose_a, const Swarm::Pose & cam_pose_b) const;
    std::vector<cv::Point2f> predictLandmarks(const VisualImageDesc & img_desc_a, 
            const Swarm::Pose & cam_pose_a, const Swarm::Pose & cam_pose_b, bool use_extrinsic=false) const;
public:
    D2FeatureTracker(D2FTConfig config);
    bool trackLocalFrames(VisualImageDescArray & frames);
    bool trackRemoteFrames(VisualImageDescArray & frames);
    void updatebySldWin(const std::vector<VINSFrame*> sld_win);
    void updatebyLandmarkDB(const std::map<LandmarkIdType, LandmarkPerId> & vins_landmark_db);
    std::vector<camodocal::CameraPtr> cams;
};


} 