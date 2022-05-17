#pragma once

#include "d2frontend_params.h"
#include "loop_cam.h"
#include "d2landmark_manager.h"
#include <unordered_map>

using namespace Eigen;

#define MAX_FEATURE_NUM 10000000

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
    std::string output_folder = "/root/output/";
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

class D2FeatureTracker {
    D2FTConfig _config;
    VisualImageDescArray current_keyframe;
    LandmarkManager * lmanager = nullptr;
    int keyframe_count = 0;
    int frame_count = 0;
    bool inited = false;
    std::map<int, LKImageInfo> prev_lk_info; //frame.camera_index->image
    LandmarkPerFrame createLKLandmark(const VisualImageDesc & frame, cv::Point2f pt, LandmarkIdType landmark_id = -1);

    TrackReport trackLK(VisualImageDesc & frame);
    TrackReport track(const VisualImageDesc & left_frame, VisualImageDesc & right_frame);
    TrackReport trackLK(const VisualImageDesc & frame, VisualImageDesc & right_frame);
    TrackReport track(VisualImageDesc & frame);
    TrackReport trackRemote(VisualImageDesc & frame);
    void processKeyframe(VisualImageDescArray & frames);
    bool isKeyframe(const TrackReport & reports);
    void draw(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report) const;
    void draw(VisualImageDesc & lframe, VisualImageDesc & rframe, bool is_keyframe, const TrackReport & report) const;
    cv::Mat drawToImage(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report, bool is_right=false) const;
    std::unordered_map<LandmarkIdType, LandmarkIdType> remote_to_local; // Remote landmark id to local;
public:
    D2FeatureTracker(D2FTConfig config):
        _config(config)
    {
        lmanager = new LandmarkManager;
    }

    bool trackLocalFrames(VisualImageDescArray & frames);
    bool trackRemoteFrames(VisualImageDescArray & frames);
    
    std::vector<camodocal::CameraPtr> cams;
};

void matchLocalFeatures(const std::vector<cv::Point2f> & pts_up, const std::vector<cv::Point2f> & pts_down, 
        const std::vector<float> & _desc_up, const std::vector<float> & _desc_down, 
        std::vector<int> & ids_down_to_up);
void detectPoints(const cv::Mat & img, std::vector<cv::Point2f> & n_pts, std::vector<cv::Point2f> & cur_pts, int require_pts);
std::vector<cv::Point2f> opticalflowTrack(const cv::Mat & cur_img, const cv::Mat & prev_img, std::vector<cv::Point2f> & prev_pts, 
                        std::vector<LandmarkIdType> & ids);

} 