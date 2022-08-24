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
struct QuadCamDepthEstConfig
{
    bool enable_depth01 = false;
    bool enable_depth12 = true;
    bool enable_depth23 = true;
    bool enable_depth30 = true;
};

struct VirtualStereoConfig {
    int minDisparity = 1;
    int numDisparities = 64;
    int blockSize = 8;
    int P1 = 8*9*9;
    int P2 = 32*9*9;
    int disp12MaxDiff = 1;
    int preFilterCap = 63;
    int uniquenessRatio = 5;
    int speckleWindowSize = 100;
    int speckleRange = 2;
    int mode = 0;
};


class VirtualStereo {
protected:
    int idx_a = 0;
    int idx_b = 1;
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
    //Rectify the images from pinhole images.
public:
    std::vector<cv::cuda::GpuMat> rectifyImage(const cv::Mat & left, const cv::Mat & right);
    cv::Mat estimateDisparityOCV(const cv::Mat & left, const cv::Mat & right);
    VirtualStereo(int _idx_a, int _idx_b, 
            const Swarm::Pose & _cam_pose_left, 
            const Swarm::Pose & _cam_pose_right, 
            D2Common::FisheyeUndist* _undist_left,
            D2Common::FisheyeUndist* _undist_right,
            int _undist_id_l, 
            int _undist_id_r);
};

class QuadCamDepthEst {
    std::vector<Swarm::Pose> raw_cam_poses;
    std::vector<VirtualStereo> virtual_stereo;
    std::vector<D2Common::FisheyeUndist*> undistortors;
public:
    void inputImages(std::vector<cv::Mat> input_imgs);
};
}