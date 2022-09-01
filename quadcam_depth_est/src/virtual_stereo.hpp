#include <swarm_msgs/Pose.h>
#include <opencv2/cudaimgproc.hpp>
namespace camodocal {
class Camera;
typedef boost::shared_ptr< Camera > CameraPtr;
}

namespace D2Common {
class FisheyeUndist;
}

namespace D2QuadCamDepthEst {
class HitnetONNX;

struct VirtualStereoConfig {
    bool use_hitnet = true;
    int minDisparity = 1;
    int numDisparities = 64;
    int blockSize = 9;
    int P1 = 8*9*9;
    int P2 = 32*9*9;
    int disp12MaxDiff = 0;
    int preFilterCap = 63;
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 32;
    int mode = 0;
};


class VirtualStereo {
protected:
    Swarm::Pose pose_left, pose_right;
    Swarm::Pose rect_pose_left, rect_pose_right;
    double baseline = 0.0;
    cv::Mat lmap_1, lmap_2, rmap_1, rmap_2;
    cv::cuda::GpuMat cuda_lmap_1, cuda_lmap_2, cuda_rmap_1, cuda_rmap_2;
    D2Common::FisheyeUndist* undist_left = nullptr, *undist_right = nullptr;
    int undist_id_l = 0;
    int undist_id_r = 1;
    cv::Mat R, T, R1, R2, T1, T2, Q;
    cv::Size img_size;
    VirtualStereoConfig config;
    HitnetONNX* hitnet = nullptr;
    cv::Rect roi_l;
    cv::Rect roi_r;
    //Rectify the images from pinhole images.
public:
    bool enable_texture = true;
    int cam_idx_a = 0;
    int cam_idx_b = 1;
    Swarm::Pose extrinsic;
    std::vector<cv::cuda::GpuMat> rectifyImage(const cv::Mat & left, const cv::Mat & right);
    cv::Mat estimateDisparityOCV(const cv::Mat & left, const cv::Mat & right);
    cv::Mat estimateDisparity(const cv::Mat & left, const cv::Mat & right);
    std::pair<cv::Mat, cv::Mat> estimateDisparityViaRaw(const cv::Mat & left, const cv::Mat & right, const cv::Mat & left_color, bool show = false);
    std::pair<cv::Mat, cv::Mat> estimatePointsViaRaw(const cv::Mat & left, const cv::Mat & right, const cv::Mat & left_color, bool show = false);
    VirtualStereo(int _idx_a, int _idx_b, 
            const Swarm::Pose & baseline, 
            D2Common::FisheyeUndist* _undist_left,
            D2Common::FisheyeUndist* _undist_right,
            int _undist_id_l, 
            int _undist_id_r, HitnetONNX* _hitnet);
};
}