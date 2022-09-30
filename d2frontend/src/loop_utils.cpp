#include <d2frontend/utils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <d2common/d2basetypes.h>
#include <d2common/utils.hpp>
#include <d2frontend/d2frontend_params.h>

using namespace std::chrono; 
using namespace D2Common;
using D2Common::Utility::TicToc;

#define PYR_LEVEL 3
#define WIN_SIZE cv::Size(21, 21)

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
    Eigen::Matrix3d R_w_c = p.R();
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

cv::Point2f rotate_pt_norm2d(cv::Point2f pt, Eigen::Quaterniond q) {
    Eigen::Vector3d pt3d(pt.x, pt.y, 1);
    pt3d = q * pt3d;

    if (pt3d.z() < 1e-3 && pt3d.z() > 0) {
        pt3d.z() = 1e-3;
    }

    if (pt3d.z() > -1e-3 && pt3d.z() < 0) {
        pt3d.z() = -1e-3;
    }

    return cv::Point2f(pt3d.x()/ pt3d.z(), pt3d.y()/pt3d.z());
}


bool inBorder(const cv::Point2f &pt, cv::Size shape)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < shape.width - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < shape.height - BORDER_SIZE;
}

std::vector<cv::DMatch> matchKNN(const cv::Mat & desc_a, const cv::Mat & desc_b, double knn_match_ratio) {
    //Match descriptors with OpenCV knnMatch
    std::vector<std::vector<cv::DMatch>> matches;
    cv::BFMatcher bfmatcher(cv::NORM_L2);
    bfmatcher.knnMatch(desc_a, desc_b, matches, 2);
    std::vector<cv::DMatch> good_matches;
    for (auto & match : matches) {
        if (match.size() < 2) {
            continue;
        }
        if (match[0].distance < knn_match_ratio * match[1].distance) {
            good_matches.push_back(match[0]);
        }
    }
    return good_matches;
}


std::vector<cv::Point2f> opticalflowTrack(const cv::Mat & cur_img, const cv::Mat & prev_img, std::vector<cv::Point2f> & prev_pts, 
        std::vector<LandmarkIdType> & ids, TrackLRType type) {
    if (prev_pts.size() == 0) {
        return std::vector<cv::Point2f>();
    }
    TicToc tic;
    std::vector<uchar> status;
    std::vector<cv::Point2f> cur_pts;
    float move_cols = cur_img.cols*90.0/params->undistort_fov; //slightly lower than 0.5 cols when fov=200

    if (prev_pts.size() == 0) {
        return std::vector<cv::Point2f>();
    }

    if (type == WHOLE_IMG_MATCH) {
        cur_pts = prev_pts;
    } else  {
        status.resize(prev_pts.size());
        std::fill(status.begin(), status.end(), 0);
        if (type == LEFT_RIGHT_IMG_MATCH) {
            for (unsigned int i = 0; i < prev_pts.size(); i++) {
                auto pt = prev_pts[i];
                if (pt.x < cur_img.cols - move_cols) {
                    pt.x += move_cols;
                    status[i] = 1;
                    cur_pts.push_back(pt);
                }
            }
        } else {
            for (unsigned int i = 0; i < prev_pts.size(); i++) {
                auto pt = prev_pts[i];
                if (pt.x >= move_cols) {
                    pt.x -= move_cols;
                    status[i] = 1;
                    cur_pts.push_back(pt);
                }
            }
        }
        reduceVector(prev_pts, status);
        reduceVector(ids, status);
    }
    status.resize(0);
    if (cur_pts.size() == 0) {
        return std::vector<cv::Point2f>();
    }
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, WIN_SIZE, PYR_LEVEL, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<uchar> reverse_status;
    std::vector<cv::Point2f> reverse_pts = cur_pts;
    for (unsigned int i = 0; i < prev_pts.size(); i++) {
        auto & pt = reverse_pts[i];
        if (type == LEFT_RIGHT_IMG_MATCH && status[i] == 1) {
            pt.x -= move_cols;
        }
        if (type == RIGHT_LEFT_IMG_MATCH && status[i] == 1) {
            pt.x += move_cols;
        }
    }
    cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, WIN_SIZE, PYR_LEVEL, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
    // if (type == D2FeatureTracker::LEFT_RIGHT_IMG_MATCH) {
    //     cv::Mat show;
    //     cv::vconcat(prev_img, cur_img, show);
    //     for (unsigned int i = 0; i < cur_pts.size(); i++) {
    //         //Draw arrows on the flow field if status[i]
    //         if (status[i] && reverse_status[i] && cv::norm(prev_pts[i] - reverse_pts[i]) <= 0.5) {
    //             cv::Point2f prev_pt = prev_pts[i];
    //             cv::Point2f cur_pt = cur_pts[i];
    //             cv::Point2f reverse_pt = reverse_pts[i];
    //             // cv::Point2f reverse_diff = reverse_pt - cur_pt;
    //             cv::arrowedLine(show, prev_pt, cur_pt, cv::Scalar(0, 255, 0), 2);
    //             cv::arrowedLine(show, cur_pt, reverse_pt, cv::Scalar(0, 0, 255), 2);
    //         }
    //     }
    //     cv::imshow("opticalflowTrack", show);
    // }
    for(size_t i = 0; i < status.size(); i++)
    {
        if(status[i] && reverse_status[i] && cv::norm(prev_pts[i] - reverse_pts[i]) <= 0.5)
        {
            status[i] = 1;
        }
        else
            status[i] = 0;
    }

    for (int i = 0; i < int(cur_pts.size()); i++){
        if (status[i] && !inBorder(cur_pts[i], cur_img.size())) {
            status[i] = 0;
        }
    }   
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(ids, status);
    return cur_pts;
} 


void detectPoints(const cv::Mat & img, std::vector<cv::Point2f> & n_pts, std::vector<cv::Point2f> & cur_pts, int require_pts) {
    int lack_up_top_pts = require_pts - static_cast<int>(cur_pts.size());
    cv::Mat mask;
    if (params->enable_perf_output) {
        ROS_INFO("Lost %d pts; Require %d will detect %d", lack_up_top_pts, require_pts, lack_up_top_pts > require_pts/4);
    }
    if (lack_up_top_pts > require_pts/4) {
        cv::Mat d_prevPts;
        cv::goodFeaturesToTrack(img, d_prevPts, lack_up_top_pts, 0.01, params->feature_min_dist, mask);
        std::vector<cv::Point2f> n_pts_tmp;
        // std::cout << "d_prevPts size: "<< d_prevPts.size()<<std::endl;
        if(!d_prevPts.empty()) {
            n_pts_tmp = cv::Mat_<cv::Point2f>(cv::Mat(d_prevPts));
        }
        else {
            n_pts_tmp.clear();
        }
        n_pts.clear();
        std::vector<cv::Point2f> all_pts = cur_pts;
        for (auto & pt : n_pts_tmp) {
            bool has_nearby = false;
            for (auto &pt_j: all_pts) {
                if (cv::norm(pt-pt_j) < params->feature_min_dist) {
                    has_nearby = true;
                    break;
                }
            }
            if (!has_nearby) {
                n_pts.push_back(pt);
                all_pts.push_back(pt);
            }
        }
    }
    else {
        n_pts.clear();
    }
}  

}