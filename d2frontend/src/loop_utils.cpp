#include <d2frontend/utils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std::chrono; 

namespace D2FrontEnd {
cv::Mat getImageFromMsg(const sensor_msgs::CompressedImageConstPtr &img_msg, int flag) {
    return cv::imdecode(img_msg->data, flag);
}

cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::Image &img_msg)
{
    cv_bridge::CvImagePtr ptr;
    // std::cout << img_msg->encoding << std::endl;
    if (img_msg.encoding == "8UC1" || img_msg.encoding == "mono8")
    {
        ptr = cv_bridge::toCvCopy(img_msg, "8UC1");
    } else if (img_msg.encoding == "16UC1" || img_msg.encoding == "mono16") {
        ptr = cv_bridge::toCvCopy(img_msg, "16UC1");
        ptr->image.convertTo(ptr->image, CV_8UC1, 1.0/256.0);
    } else {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);        
    }
    return ptr;
}

cv_bridge::CvImagePtr getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImagePtr ptr;
    // std::cout << img_msg->encoding << std::endl;
    if (img_msg->encoding == "8UC1" || img_msg->encoding == "mono8")
    {
        ptr = cv_bridge::toCvCopy(img_msg, "8UC1");
    } else if (img_msg->encoding == "16UC1" || img_msg->encoding == "mono16") {
        ptr = cv_bridge::toCvCopy(img_msg, "16UC1");
        ptr->image.convertTo(ptr->image, CV_8UC1, 1.0/256.0);
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

cv::Vec3b extractColor(const cv::Mat &img, cv::Point2f p) {
    cv::Vec3b color;
    if (img.channels() == 3) {
        color = img.at<cv::Vec3b>(p);
    } else {
        auto grayscale = img.at<uchar>(p);
        color = cv::Vec3b(grayscale, grayscale, grayscale);
    }
    return color;
}


#define MAXBUFSIZE 100000
Eigen::MatrixXf load_csv_mat_eigen(std::string csv) {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(csv);
    std::string line;

    while (getline(infile, line))
    {
        int temp_cols = 0;
        std::stringstream          lineStream(line);
        std::string                cell;

        while (std::getline(lineStream, cell, ','))
        {
            buff[rows * cols + temp_cols] = std::stod(cell);
            temp_cols ++;
        }

        rows ++;
        if (cols > 0) {
            assert(cols == temp_cols && "Matrix must have same cols on each rows!");
        } else {
            cols = temp_cols;
        }
    }

    infile.close();

    Eigen::MatrixXf result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
}

Eigen::VectorXf load_csv_vec_eigen(std::string csv) {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    std::ifstream infile;
    infile.open(csv);
    while (! infile.eof())
    {
        std::string line;
        getline(infile, line);

        int temp_cols = 0;
        std::stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    Eigen::VectorXf result(rows,cols);
    for (int i = 0; i < rows; i++)
            result(i) = buff[ i ];

    return result;
}

}