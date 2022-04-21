#pragma once

#include "d2frontend_params.h"
#include "loop_cam.h"

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
    double ransacReprojThreshold = 10;
    std::string output_folder = "/root/output/";
};

struct TrackReport {
    double sum_parallex = 0.0;
    int parallex_num = 0;
    int long_track_num = 0.0;
    int unmatched_num = 0.0;
    double ft_time = 0.0;

    void compose(const TrackReport & report) {
        sum_parallex += report.sum_parallex;
        parallex_num += report.parallex_num;
        long_track_num += report.long_track_num;
        unmatched_num += report.unmatched_num;
    }

    double meanParallex() const {
        return sum_parallex/parallex_num;
    }
};

class LandmarkManager {
protected:
    std::map<LandmarkIdType, LandmarkPerId> landmark_db;
    int count = 0;
public:
    virtual int addLandmark(const LandmarkPerFrame & lm);
    virtual void updateLandmark(const LandmarkPerFrame & lm);
    LandmarkPerId & at(int i) {
        return landmark_db.at(i);
    }
};

class D2FeatureTracker {
    D2FTConfig _config;
    VisualImageDescArray current_keyframe;
    LandmarkManager * lmanager = nullptr;
    int keyframe_count = 0;
    int frame_count = 0;
    bool inited = false;
public:
    D2FeatureTracker(D2FTConfig config):
        _config(config)
    {
        lmanager = new LandmarkManager;
    }

    bool track(VisualImageDescArray & frames);
    TrackReport track(VisualImageDesc & frame);
    void processKeyframe(VisualImageDescArray & frames);
    bool isKeyframe(const TrackReport & reports);
    void draw(VisualImageDesc & frame, bool is_keyframe, const TrackReport & report);
    void matchLocalFeatures(const std::vector<cv::Point2f> & pts_up, const std::vector<cv::Point2f> & pts_down, 
        std::vector<float> & _desc_up, std::vector<float> & _desc_down, 
        std::vector<int> & ids_down_to_up) const;
};


} 