#pragma once
#include <sensor_msgs/PointCloud.h>
#include <pcl_ros/point_cloud.h>
#include <opencv2/opencv.hpp>

namespace D2QuadCamDepthEst {
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;


inline void addtoPCL(PointCloud & pcl, Vector3f point) {
    pcl::PointXYZ p(point(0), point(1), point(2));
    pcl.points.push_back(p);
}

inline void addtoPCL(PointCloudRGB & pcl, Vector3f point) {
    pcl::PointXYZRGB p;
    p.x = point(0);
    p.y = point(1);
    p.z = point(2);
    pcl.points.push_back(p);
}

inline void addtoPCL(PointCloud & pcl, Vector3f point, cv::Vec3b color) {
    pcl::PointXYZ p(point(0), point(1), point(2));
    pcl.points.push_back(p);
}

inline void addtoPCL(PointCloudRGB & pcl, Vector3f point, cv::Vec3b color) {
    pcl::PointXYZRGB p;
    p.x = point(0);
    p.y = point(1);
    p.z = point(2);
    p.r = color[2], 255;
    p.g = color[1], 255;
    p.b = color[0], 255;
    pcl.points.push_back(p);
}

inline void addtoPCL(PointCloudRGB &pcl, Vector3f point, uchar grayscale) {
    pcl::PointXYZRGB p;
    p.x = point(0);
    p.y = point(1);
    p.z = point(2);
    p.r = grayscale;
    p.g = grayscale;
    p.b = grayscale;
    pcl.points.push_back(p);
}

template<typename PointType>
void addPointsToPCL(const cv::Mat & pts3d, const cv::Mat & color, Swarm::Pose pose, 
        pcl::PointCloud<PointType> & pcl, int step, double min_z, double max_z) {
    bool rgb_color = color.channels() == 3;
    Matrix3f R = pose.R().template cast<float>();
    Vector3f t = pose.pos().template cast<float>();
    for(int v = 0; v < pts3d.rows; v += step) {
        for(int u = 0; u < pts3d.cols; u += step) {
            cv::Vec3f vec = pts3d.at<cv::Vec3f>(v, u);
            Vector3f pts_i(vec[0], vec[1], vec[2]);
            if (pts_i.z() < max_z && pts_i.z() > min_z) {
                Vector3f w_pts_i = R * pts_i + t;
                // Vector3d w_pts_i = pts_i;
                if (color.empty()) {
                    addtoPCL(pcl, w_pts_i);
                } else {
                    int32_t rgb_packed;
                    if(rgb_color) {
                        if (color.type() == CV_8UC3) {
                            const cv::Vec3b& bgr = color.at<cv::Vec3b>(v, u);
                            addtoPCL(pcl, w_pts_i, bgr);
                        } else if(color.type() == CV_32FC3) {
                            const cv::Vec3f& bgr = color.at<cv::Vec3f>(v, u);
                            cv::Vec3b bgr_8u;
                            bgr_8u[0] = std::min((int)bgr[0], 255);
                            bgr_8u[1] = std::min((int)bgr[1], 255);
                            bgr_8u[2] = std::min((int)bgr[2], 255);
                            addtoPCL(pcl, w_pts_i, bgr);
                        }
                    } else {
                        const uchar& bgr = color.at<uchar>(v, u);
                        addtoPCL(pcl, w_pts_i, bgr);
                    }
                }
            }
        }
    }
}
}