#include <swarm_loop/utils.h>
#include <opencv/cv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std::chrono; 

cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg, int flag) {
    return cv::imdecode(img_msg->data, flag);
}

cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::Image &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    // std::cout << img_msg->encoding << std::endl;
    if (img_msg.encoding == "8UC1" || img_msg.encoding == "mono8")
    {
        ptr = cv_bridge::toCvCopy(img_msg, "8UC1");
    } else if (img_msg.encoding == "16UC1") {
        ptr = cv_bridge::toCvCopy(img_msg, "16UC1");
    } else {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);        
    }
    return ptr;
}

cv_bridge::CvImageConstPtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    // std::cout << img_msg->encoding << std::endl;
    if (img_msg->encoding == "8UC1" || img_msg->encoding == "mono8")
    {
        ptr = cv_bridge::toCvCopy(img_msg, "8UC1");
    } else if (img_msg->encoding == "16UC1") {
        ptr = cv_bridge::toCvCopy(img_msg, "16UC1");
    } else {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);        
    }
    return ptr;
}

Swarm::Pose AffineRestoCamPose(Eigen::Matrix4d affine) {
    Eigen::Matrix3d R;
    Eigen::Vector3d T;

    R = affine.block<3, 3>(0, 0);
    T = affine.block<3, 1>(0, 3);
    
    R = (R.normalized()).transpose();
    T = R *(-T);

    std::cout << "R of affine\n" << R << std::endl;
    std::cout << "T of affine\n" << T << std::endl;
    std::cout << "RtR\n" << R.transpose()*R << std::endl;
    return Swarm::Pose(R, T);
}

void PnPInitialFromCamPose(const Swarm::Pose &p, cv::Mat & rvec, cv::Mat & tvec) {
    Eigen::Matrix3d R_w_c = p.att().toRotationMatrix();
    Eigen::Matrix3d R_inital = R_w_c.inverse();
    Eigen::Vector3d T_w_c = p.pos();
    cv::Mat tmp_r;
    Eigen::Vector3d P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, tvec);
}

Swarm::Pose PnPRestoCamPose(cv::Mat rvec, cv::Mat tvec) {
    cv::Mat r;
    cv::Rodrigues(rvec, r);
    Eigen::Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Eigen::Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(tvec, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    return Swarm::Pose(R_w_c_old, T_w_c_old);
}
