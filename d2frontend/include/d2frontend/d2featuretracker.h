#pragma once

#include "d2frontend_params.h"
#include "loop_cam.h"
#include "d2landmark_manager.h"
#include <unordered_map>
#include <mutex>

using namespace Eigen;

namespace D2FrontEnd {
struct D2FTConfig {
    bool show_feature_id = true;
    int long_track_thres = 20;
    int long_track_frames = 4;
    int last_track_thres = 20;
    double new_feature_thres = 0.5;
    double parallex_thres = 10.0/460.0;
    int min_keyframe_num = 2;
    bool write_to_file = false;
    bool check_homography = false;
    bool enable_lk_optical_flow = true;
    double ransacReprojThreshold = 10;
    double max_pts_velocity_time=0.3;
    int remote_min_match_num = 30;
    bool double_counting_common_feature = false;
    bool enable_superglue_local = false;
    bool enable_superglue_remote = false;
    std::string output_folder = "/root/output/";
    std::string superglue_model_path;
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

struct LKImageInfo {
    FrameIdType frame_id;
    std::vector<cv::Point2f> lk_pts;
    std::vector<LandmarkIdType> lk_ids;
    cv::Mat image;
};

class SuperGlueOnnx;

class D2FeatureTracker {
public:
    enum TrackLRType {
        WHOLE_IMG_MATCH = 0,
        LEFT_RIGHT_IMG_MATCH,
        RIGHT_LEFT_IMG_MATCH
    };
protected:
    D2FTConfig _config;
    std::vector<VisualImageDescArray> current_keyframes;
    LandmarkManager * lmanager = nullptr;
    int keyframe_count = 0;
    int frame_count = 0;
    bool inited = false;
    std::map<int, LKImageInfo> prev_lk_info; //frame.camera_index->image
    std::pair<bool, LandmarkPerFrame> createLKLandmark(const VisualImageDesc & frame, cv::Point2f pt, LandmarkIdType landmark_id = -1);

    TrackReport trackLK(VisualImageDesc & frame);
    TrackReport track(const VisualImageDesc & left_frame, VisualImageDesc & right_frame, bool enable_lk=true, TrackLRType type=WHOLE_IMG_MATCH);
    TrackReport trackLK(const VisualImageDesc & frame, VisualImageDesc & right_frame, TrackLRType type=WHOLE_IMG_MATCH);
    TrackReport track(VisualImageDesc & frame);
    TrackReport trackRemote(VisualImageDesc & frame, bool skip_whole_frame_match=false);
    void processKeyframe(VisualImageDescArray & frames);
    bool isKeyframe(const TrackReport & reports);
    Vector3d extractPointVelocity(const LandmarkPerFrame & lpf) const;
    std::pair<bool, LandmarkPerFrame> getPreviousLandmarkFrame(const LandmarkPerFrame & lpf) const;

    void draw(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report) const;
    void draw(VisualImageDesc & lframe, VisualImageDesc & rframe, bool is_keyframe, const TrackReport & report) const;
    void draw(std::vector<VisualImageDesc> frames, bool is_keyframe, const TrackReport & report) const;
    void drawRemote(VisualImageDesc & frame, const TrackReport & report) const;
    void cvtRemoteLandmarkId(VisualImageDesc & frame) const;
    cv::Mat drawToImage(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report, bool is_right=false, bool is_remote=false) const;
    std::unordered_map<LandmarkIdType, LandmarkIdType> remote_to_local; // Remote landmark id to local;
    std::unordered_map<LandmarkIdType, std::unordered_map<int, LandmarkIdType>> local_to_remote; // local landmark id to remote drone and id;
    typedef std::lock_guard<std::recursive_mutex> Guard;
    std::recursive_mutex track_lock;
    SuperGlueOnnx * superglue = nullptr;
    bool matchLocalFeatures(const VisualImageDesc & img_desc_a, const VisualImageDesc & img_desc_b, std::vector<int> & ids_down_to_up, 
            bool enable_superglue=true, TrackLRType type=WHOLE_IMG_MATCH);
public:
    D2FeatureTracker(D2FTConfig config);
    bool trackLocalFrames(VisualImageDescArray & frames);
    bool trackRemoteFrames(VisualImageDescArray & frames);
    
    std::vector<camodocal::CameraPtr> cams;
};


void detectPoints(const cv::Mat & img, std::vector<cv::Point2f> & n_pts, std::vector<cv::Point2f> & cur_pts, int require_pts);
std::vector<cv::Point2f> opticalflowTrack(const cv::Mat & cur_img, const cv::Mat & prev_img, std::vector<cv::Point2f> & prev_pts, 
                        std::vector<LandmarkIdType> & ids, D2FeatureTracker::TrackLRType type=D2FeatureTracker::WHOLE_IMG_MATCH);

} 