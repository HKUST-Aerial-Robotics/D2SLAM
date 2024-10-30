#pragma once

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/CylindricalCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>

#include <d2common/utils.hpp>

#include "cv_bridge/cv_bridge.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"

#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace D2Common {

using D2Common::Utility::TicToc;

#define DEG_TO_RAD (M_PI / 180.0)
#define REMAP_FUNC cv::INTER_LINEAR
// #define REMAP_FUNC cv::INTER_NEAREST
class FisheyeUndist {
    typedef std::vector<std::pair<cv::Mat, cv::Mat>> MapArray;
    camodocal::CameraPtr cam;

    std::vector<cv::cuda::GpuMat> undistMapsGPUX;
    std::vector<cv::cuda::GpuMat> undistMapsGPUY;

   public:
    enum UndistortType {
        UndistortPinhole5,  // 1 center image + 4 side images/
        UndistortCylindrical,
        UndistortPinhole2  // two images for stereo, maybe slightly overlapped,
                           // this is for quadcam depth generation.
    };
    MapArray undistMaps;

    cv::Mat fisheye2cam_pt;
    cv::Mat fisheye2cam_id;

    camodocal::CameraPtr cam_top;
    camodocal::CameraPtr cam_side;
    double f_side = 0;
    double f_center = 0;
    double cx_side = 0, cy_side = 0;
    int imgWidth = 0;
    double fov = 0;  // in degree
    Eigen::Vector3d cameraRotation;
    int cam_id = 0;

    double raw_width;
    double raw_height;

    int sideImgHeight = 0;

    std::vector<Eigen::Quaterniond> t;
    std::vector<cv::Mat> photometics;
    std::vector<cv::Mat> photometics_bgr;

#ifdef USE_CUDA
    std::vector<cv::cuda::GpuMat> photometics_gpu;
    std::vector<cv::cuda::GpuMat> photometics_gpu_bgr;
    cv::cuda::GpuMat img_cuda;
#endif

    FisheyeUndist(const std::string &camera_config_file, int _id, double _fov,
                  int imgWidth = 600)
        : imgWidth(imgWidth),
          fov(_fov),
          cameraRotation(0, 0, 0),
          cam_id(_id) {
        cam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(
            camera_config_file);
        raw_width = cam->imageWidth();
        raw_height = cam->imageHeight();
        fisheye2cam_pt = cv::Mat::zeros(raw_width, raw_height, CV_32FC2);
        fisheye2cam_id = cv::Mat::ones(raw_width, raw_height, CV_8UC1);
        fisheye2cam_id = fisheye2cam_id * 255;
        undistMaps = generateAllUndistMap(cam, cameraRotation, imgWidth, fov);
        // ROS_INFO("undismap size %ld", undistMaps.size());
#ifdef USE_CUDA
            for (auto mat : undistMaps) {
                cv::Mat maps[2];
                cv::split(mat.first, maps);
                undistMapsGPUX.push_back(cv::cuda::GpuMat(maps[0]));
                undistMapsGPUY.push_back(cv::cuda::GpuMat(maps[1]));
            }
#endif
    }

    FisheyeUndist(camodocal::CameraPtr cam, int _id, double _fov,
                  UndistortType mode = UndistortPinhole5, int imgWidth = 600,
                  int imgHeight = 200, cv::Mat photomertic=cv::Mat())
        : imgWidth(imgWidth),
          fov(_fov),
          cameraRotation(0, 0, 0),
          cam_id(_id) {
        raw_width = cam->imageWidth();
        raw_height = cam->imageHeight();
        fisheye2cam_pt = cv::Mat::zeros(raw_width, raw_height, CV_32FC2);
        fisheye2cam_id = cv::Mat::ones(raw_width, raw_height, CV_8UC1);
        fisheye2cam_id = fisheye2cam_id * 255;
        if (mode == UndistortPinhole5) {
            undistMaps =
                generateAllUndistMap(cam, cameraRotation, imgWidth, fov);
        } else if (mode == UndistortCylindrical) {
            undistMaps = generateCylinderMap(cam, cameraRotation, imgWidth,
                                             imgHeight, fov);
        } else if (mode == UndistortPinhole2) {
            undistMaps = generatePinhole2Map(cam, cameraRotation, imgWidth,
                                             imgHeight, fov);
        }
        // ROS_INFO("undismap size %ld", undistMaps.size());
#if USE_CUDA
            for (auto mat : undistMaps) {
                cv::Mat maps[2];
                cv::split(mat.first, maps);
                undistMapsGPUX.push_back(cv::cuda::GpuMat(maps[0]));
                undistMapsGPUY.push_back(cv::cuda::GpuMat(maps[1]));
            }
#endif
        if (!photomertic.empty()) {
            auto _photometics = undist_all(photomertic, true);
            photometics = _photometics;
#ifdef USE_CUDA
            auto _photometics_gpu = undist_all_cuda(photomertic, true);
            photometics_gpu = _photometics_gpu;
            for (unsigned int i = 0; i < _photometics.size(); i++) {
                cv::Mat bgr;
                cv::cvtColor(_photometics[i], bgr, cv::COLOR_GRAY2BGR);
                photometics_bgr.push_back(bgr);
                cv::cuda::GpuMat bgr_gpu;
                cv::cuda::cvtColor(_photometics_gpu[i], bgr_gpu,
                                   cv::COLOR_GRAY2BGR);
                photometics_gpu_bgr.push_back(bgr_gpu);
            }
#endif
        } else {
            printf("no photometric calibration file found\n");
        }
    }

#ifdef USE_CUDA
    cv::cuda::GpuMat undist_id_cuda(cv::Mat image, int _id, bool calib_photometric=false) {
        cv::cuda::GpuMat img_cuda(image);
        return undist_id_cuda(img_cuda, _id, calib_photometric);
        assert(false && "CUDA not enabled");
    }

    cv::cuda::GpuMat undist_id_cuda(cv::cuda::GpuMat img_cuda, int _id, bool calib_photometric=false) {
        cv::cuda::GpuMat output;
        cv::cuda::remap(img_cuda, output, undistMapsGPUX[_id],
                        undistMapsGPUY[_id], REMAP_FUNC);
        if (photometics_gpu.size() > 0 && calib_photometric) {
            if (output.channels() == 3) {
                output.convertTo(output, CV_32FC3);
                cv::cuda::multiply(output, photometics_gpu_bgr[_id], output);
                output.convertTo(output, CV_8UC3);
            } else {
                output.convertTo(output, CV_32FC1);
                cv::cuda::multiply(output, photometics_gpu[_id], output);
                // cv::cuda::threshold(output, output, 255, 255, cv::THRESH_TRUNC);
                output.convertTo(output, CV_8U);
            }
        }
        return output;
        assert(false && "CUDA not enabled");
    }

    std::vector<cv::Mat> undist_all_cuda_cpu(
        const cv::Mat &image, bool use_rgb = false,
        std::vector<bool> mask = std::vector<bool>(0)) {
        TicToc up;
        bool has_mask = mask.size() == undistMaps.size();
        if (use_rgb) {
            img_cuda.upload(image);
        } else {
            cv::Mat _tmp;
            cv::cvtColor(image, _tmp, cv::COLOR_BGR2GRAY);
            img_cuda.upload(_tmp);
        }

        std::cout << "Upload cost " << up.toc() << std::endl;

        std::vector<cv::Mat> ret;
        for (unsigned int i = 0; i < undistMaps.size(); i++) {
            cv::Mat tmp;
            if (!has_mask || (has_mask && mask[i])) {
                cv::cuda::GpuMat output;
                TicToc remap;
                cv::cuda::remap(img_cuda, output, undistMapsGPUX[i],
                                undistMapsGPUY[i], REMAP_FUNC);
                if (photometics_gpu.size() > 0) {
                    if (image.channels() == 3) {
                        cv::cuda::multiply(output, photometics_gpu_bgr[i],
                                           output);
                    } else {
                        cv::cuda::multiply(output, photometics_gpu[i], output);
                    }
                }
                std::cout << "Remap cost " << remap.toc() << std::endl;
                TicToc down;
                output.download(tmp);
                std::cout << "Download cost " << down.toc() << std::endl;
            }
            ret.push_back(tmp);
        }
        return ret;
    }

    std::vector<cv::cuda::GpuMat> undist_all_cuda(
        const cv::Mat &image, bool use_rgb = false,
        std::vector<bool> mask = std::vector<bool>(0)) {
        cv::cuda::GpuMat img_cuda;
        bool has_mask = mask.size() == undistMaps.size();
        if (use_rgb) {
            img_cuda.upload(image);
        } else {
            cv::Mat _tmp;
            cv::cvtColor(image, _tmp, cv::COLOR_BGR2GRAY);
            img_cuda.upload(_tmp);
        }

        std::vector<cv::cuda::GpuMat> ret;
        for (unsigned int i = 0; i < undistMaps.size(); i++) {
            cv::cuda::GpuMat output;
            if (!has_mask || (has_mask && mask[i])) {
                cv::cuda::remap(img_cuda, output, undistMapsGPUX[i],
                                undistMapsGPUY[i], REMAP_FUNC);
                if (photometics_gpu.size() > 0) {
                    if (image.channels() == 3) {
                        cv::cuda::multiply(output, photometics_gpu_bgr[i],
                                           output);
                    } else {
                        cv::cuda::multiply(output, photometics_gpu[i], output);
                    }
                }
            }
            ret.push_back(output);
        }
        return ret;
    }
#endif

    std::vector<cv::Mat> undist_all(const cv::Mat &image, bool use_rgb = false,
                                    bool enable_top = true,
                                    bool enable_rear = true) {
        std::vector<cv::Mat> ret;
        ret.resize(undistMaps.size());
        bool disable[5] = {0};
        disable[0] = !enable_top;
        disable[5] = !enable_rear;
        if (use_rgb) {
#pragma omp parallel for num_threads(5)
            for (unsigned int i = 0; i < undistMaps.size(); i++) {
                if (!disable[i]) {
                    cv::remap(image, ret[i], undistMaps[i].first,
                              undistMaps[i].second, REMAP_FUNC);
                    if (photometics.size() > 0) {
                        cv::multiply(ret[i], photometics[i], ret[i]);
                    }
                }
            }
            return ret;
        } else {
            cv::Mat gray;
            if (image.channels() == 3)
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            else
                gray = image;
#pragma omp parallel for num_threads(5)
            for (unsigned int i = 0; i < undistMaps.size(); i++) {
                if (!disable[i]) {
                    cv::remap(gray, ret[i], undistMaps[i].first,
                              undistMaps[i].second, REMAP_FUNC);
                    if (photometics.size() > 0) {
                        cv::multiply(ret[i], photometics[i], ret[i]);
                    }
                }
            }
            return ret;
        }

        return ret;
    }

    void stereo_flatten(const cv::Mat &image1, const cv::Mat &image2,
                        FisheyeUndist *undist2, std::vector<cv::Mat> &lefts,
                        std::vector<cv::Mat> &rights, bool use_rgb = false,
                        bool enable_up_top = true, bool enable_up_rear = true,
                        bool enable_down_top = true,
                        bool enable_down_rear = true) {
        auto method = REMAP_FUNC;
        lefts.resize(5);
        rights.resize(5);
        bool disable[10] = {0};
        disable[0] = !enable_up_top;
        disable[4] = !enable_up_rear;

        disable[5] = !enable_down_top;
        disable[9] = !enable_down_rear;

        if (use_rgb) {
#pragma omp parallel for num_threads(10)
            for (unsigned int i = 0; i < 10; i++) {
                if (!disable[i]) {
                    if (i > 4) {
                        cv::remap(image2, rights[i % 5],
                                  undist2->undistMaps[i % 5].first,
                                  undist2->undistMaps[i % 5].second, method);
                    } else {
                        cv::remap(image1, lefts[i], undistMaps[i % 5].first,
                                  undistMaps[i % 5].second, method);
                    }
                }
            }
        } else {
            cv::Mat gray1, gray2;
            cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
            cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
#pragma omp parallel for num_threads(10)
            for (unsigned int i = 0; i < 10; i++) {
                if (!disable[i]) {
                    if (i > 4) {
                        cv::remap(gray2, rights[i % 5],
                                  undist2->undistMaps[i % 5].first,
                                  undist2->undistMaps[i % 5].second, method);
                    } else {
                        cv::remap(gray1, lefts[i], undistMaps[i % 5].first,
                                  undistMaps[i % 5].second, method);
                    }
                }
            }
        }
    }

    // top + side
    MapArray generateAllUndistMap(camodocal::CameraPtr p_cam,
                                  Eigen::Vector3d rotation,
                                  const unsigned &imgWidth,
                                  const double &fov  // degree
    ) {
        // ROS_INFO("Generating undistortion maps:");
        double sideVerticalFOV = (fov - 180) * DEG_TO_RAD;
        if (sideVerticalFOV < 0) sideVerticalFOV = 0;
        double centerFOV = fov * DEG_TO_RAD - sideVerticalFOV * 2;
        ROS_INFO("Build for camera %d", cam_id);
        ROS_INFO("Center FOV: %f_center", centerFOV);

        // calculate focal length of fake pinhole cameras (pixel size = 1 unit)
        f_center = (double)imgWidth / 2 / tan(centerFOV / 2);
        f_side = (double)imgWidth / 2;

        // sideImgHeight = sideVerticalFOV / centerFOV * imgWidth;
        sideImgHeight = 2 * f_side * tan(sideVerticalFOV / 2);

        ROS_INFO("Side image height: %d", sideImgHeight);
        std::vector<std::pair<cv::Mat, cv::Mat>> maps;
        maps.reserve(5);

        // test points
        Eigen::Vector3d testPoints[] = {
            Eigen::Vector3d(0, 0, 1),
            Eigen::Vector3d(1, 0, 1),
            Eigen::Vector3d(0, 1, 1),
            Eigen::Vector3d(1, 1, 1),
        };
        for (unsigned int i = 0;
             i < sizeof(testPoints) / sizeof(Eigen::Vector3d); i++) {
            Eigen::Vector2d temp;
            p_cam->spaceToPlane(testPoints[i], temp);
            // ROS_INFO("Test point %d : (%.2f,%.2f,%.2f) projected to
            // (%.2f,%.2f)", i, testPoints[i][0], testPoints[i][1],
            // testPoints[i][2], temp[0], temp[1]);
        }

        t.resize(5);
        t[0] = Eigen::Quaterniond::Identity();

        // ROS_INFO("Pinhole cameras focal length: center %f side %f", f_center,
        // f_side);

        cam_top = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera(
            "top", imgWidth, imgWidth, 0, 0, 0, 0, f_center, f_center,
            imgWidth / 2, imgWidth / 2));

        cx_side = imgWidth / 2;
        cy_side = sideImgHeight / 2;
        cam_side = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera(
            "side", imgWidth, sideImgHeight, 0, 0, 0, 0, f_side, f_side,
            imgWidth / 2, sideImgHeight / 2));

        maps.push_back(
            genOneUndistMap(0, p_cam, t[0], imgWidth, imgWidth, f_center));

        Eigen::Quaterniond t0 = t[0];
        if (cam_id == 1) {
            std::cout << "Is camera 1 will invert T" << std::endl;
            t0 = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
        };

        if (sideImgHeight > 0) {
            // facing y
            t[1] = t0 * Eigen::AngleAxis<double>(-M_PI / 2,
                                                 Eigen::Vector3d(1, 0, 0));
            maps.push_back(genOneUndistMap(1, p_cam, t[1], imgWidth,
                                           sideImgHeight, f_side));

            // turn right/left?
            t[2] = t[1] *
                   Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(2, p_cam, t[2], imgWidth,
                                           sideImgHeight, f_side));
            t[3] = t[2] *
                   Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(3, p_cam, t[3], imgWidth,
                                           sideImgHeight, f_side));
            t[4] = t[3] *
                   Eigen::AngleAxis<double>(M_PI / 2, Eigen::Vector3d(0, 1, 0));
            maps.push_back(genOneUndistMap(4, p_cam, t[4], imgWidth,
                                           sideImgHeight, f_side));
        }
        return maps;
    }

    MapArray generatePinhole2Map(camodocal::CameraPtr p_cam,
                                 Eigen::Vector3d rotation,
                                 const unsigned &imgWidth,
                                 const unsigned &imgHeight, const double &fov) {
        double pinhole_fov = fov * DEG_TO_RAD - M_PI/2;
        ROS_INFO("New pinhole2 fov: %f", pinhole_fov * 57.3);
        cx_side = imgWidth / 2;
        cy_side = imgHeight / 2;
        sideImgHeight = imgHeight;
        //Fov
        f_side = (double)imgWidth / 2 / tan(pinhole_fov / 2);
        cam_side = camodocal::PinholeCameraPtr(new camodocal::PinholeCamera(
            "side", imgWidth, sideImgHeight, 0, 0, 0, 0, f_side, f_side,
            imgWidth / 2, sideImgHeight / 2));
        t.resize(2);
        t[0] = Eigen::Quaterniond(Eigen::AngleAxis<double>(-M_PI / 4, Eigen::Vector3d(0, 1, 0)));
        t[1] = Eigen::Quaterniond(Eigen::AngleAxis<double>(M_PI / 4, Eigen::Vector3d(0, 1, 0)));
        MapArray maps;
        maps.push_back(genOneUndistMap(0, p_cam, t[0], imgWidth, sideImgHeight, f_side));
        maps.push_back(genOneUndistMap(1, p_cam, t[1], imgWidth, sideImgHeight, f_side));
        return maps;
    }
    
    // cylinder camera model.
    MapArray generateCylinderMap(camodocal::CameraPtr p_cam,
                                 Eigen::Vector3d rotation,
                                 const unsigned &imgWidth,
                                 const unsigned &imgHeight, const double &fov) {
        double cylinderFov = fov * DEG_TO_RAD;
        // ROS_INFO("Build for camera %d", cam_id);
        printf("[FisheyeUndist] Generating Cylinder maps, fov %.1f deg\n", cylinderFov * 57.3);

        f_center = imgWidth / cylinderFov;

        std::vector<std::pair<cv::Mat, cv::Mat>> maps;
        // test points
        Eigen::Vector3d testPoints[] = {
            Eigen::Vector3d(0, 0, 1),
            Eigen::Vector3d(1, 0, 1),
            Eigen::Vector3d(0, 1, 1),
            Eigen::Vector3d(1, 1, 1),
        };
        for (unsigned int i = 0;
             i < sizeof(testPoints) / sizeof(Eigen::Vector3d); i++) {
            Eigen::Vector2d temp;
            p_cam->spaceToPlane(testPoints[i], temp);
            // ROS_INFO("Test point %d : (%.2f,%.2f,%.2f) projected to
            // (%.2f,%.2f)", i,
            //         testPoints[i][0], testPoints[i][1], testPoints[i][2],
            //         temp[0], temp[1]);
        }

        t.resize(5);
        t[0] = Eigen::Quaterniond::Identity();

        // ROS_INFO("Pinhole cameras focal length: center %f side %f", f_center,
        // f_side);

        cam_top =
            camodocal::CylindricalCameraPtr(new camodocal::CylindricalCamera(
                "cylindrical", imgWidth, imgHeight, f_center, f_center,
                imgWidth / 2, imgHeight / 2));

        maps.push_back(
            genOneUndistMap(0, p_cam, cam_top, t[0], imgWidth, imgHeight));
        return maps;
    }

    std::pair<int, cv::Point2f> project_point_to_vcam_id(
        Eigen::Vector3d pts_cam) {
        // First project the point to fisheye image plane
        Eigen::Vector2d imgPoint;
        cam->spaceToPlane(pts_cam, imgPoint);
        cv::Point2f pt =
            fisheye2cam_pt.at<cv::Vec2f>(cv::Point(imgPoint.x(), imgPoint.y()));
        int id = 255;
        if (!imgPoint.hasNaN() && imgPoint.x() >= 0 &&
            imgPoint.x() < raw_width && imgPoint.y() >= 0 &&
            imgPoint.y() < raw_height) {
            fisheye2cam_id.at<uint8_t>(cv::Point(imgPoint.x(), imgPoint.y()));
        }
        if (id != 255) {
            // std::cout << "\n\nPT" << pts_cam << " IMG " << imgPoint << " ID"
            // << id << " PT " << pt << std::endl;
            return std::make_pair(id, pt);
        } else {
            // std::cout << "Map not found, try project" << std::endl;
        }

        for (int i = 0; i < 5; i++) {
            auto pts_vcam = t[i].inverse() * pts_cam;
            int width = imgWidth;
            int height = sideImgHeight;
            if (pts_vcam.z() < 0) {
                continue;
            }

            if (i == 0) {
                height = imgWidth;
                cam_top->spaceToPlane(pts_vcam, imgPoint);
            } else {
                cam_side->spaceToPlane(pts_vcam, imgPoint);
            }

            if (!imgPoint.hasNaN() && imgPoint.x() >= 0 &&
                imgPoint.x() < width && imgPoint.y() >= 0 &&
                imgPoint.y() < height) {
                id = i;
                pt.x = imgPoint.x();
                pt.y = imgPoint.y();
                break;
            }
        }

        if (id == 255) {
            // std::cout << "PT " << pts_cam << "not found in image" <<
            // std::endl;
            return std::make_pair(-1, cv::Point2f(0, 0));
        } else {
            // std::cout << "\n\nPT" << pts_cam << "IMG " << imgPoint << "ID" <<
            // id << "PT" << pt << std::endl;
            return std::make_pair(id, pt);
        }
    }

    std::pair<cv::Mat, cv::Mat> genOneUndistMap(int _id,
                                                camodocal::CameraPtr p_cam,
                                                camodocal::CameraPtr p_vcam,
                                                Eigen::Quaterniond rotation,
                                                const unsigned &imgWidth,
                                                const unsigned &imgHeight) {
        cv::Mat map = cv::Mat(imgHeight, imgWidth, CV_32FC2);
        ROS_DEBUG("Generating map of size (%d,%d)", map.size[0], map.size[1]);
        ROS_DEBUG("Perspective facing (%.2f,%.2f,%.2f)",
                  (rotation * Eigen::Vector3d(0, 0, 1))[0],
                  (rotation * Eigen::Vector3d(0, 0, 1))[1],
                  (rotation * Eigen::Vector3d(0, 0, 1))[2]);
        for (unsigned int x = 0; x < imgWidth; x++)
            for (unsigned int y = 0; y < imgHeight; y++) {
                Eigen::Vector3d objPoint;
                p_vcam->liftProjective(Eigen::Vector2d(x, y), objPoint);
                Eigen::Vector2d imgPoint;
                p_cam->spaceToPlane(objPoint, imgPoint);

                map.at<cv::Vec2f>(cv::Point(x, y)) =
                    cv::Vec2f(imgPoint.x(), imgPoint.y());
                if (!isnan(imgPoint.x()) && !isnan(imgPoint.y()) &&
                    imgPoint.x() >= 0 && imgPoint.x() <= raw_width &&
                    imgPoint.y() >= 0 && imgPoint.y() <= raw_height) {
                    auto &pt = fisheye2cam_pt.at<cv::Vec2f>(
                        cv::Point(imgPoint.x(), imgPoint.y()));
                    fisheye2cam_id.at<uint8_t>(
                        cv::Point(imgPoint.x(), imgPoint.y())) = _id;
                    pt[0] = x;
                    pt[1] = y;
                }
            }

        ROS_DEBUG("Upper corners: (%.2f, %.2f), (%.2f, %.2f)",
                  map.at<cv::Vec2f>(cv::Point(0, 0))[0],
                  map.at<cv::Vec2f>(cv::Point(0, 0))[1],
                  map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[0],
                  map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[1]);

        Eigen::Vector3d objPoint =
            rotation * Eigen::Vector3d(((double)0 - (double)imgWidth / 2),
                                       ((double)0 - (double)imgHeight / 2),
                                       f_center);
        // std::cout << objPoint << std::endl;

        objPoint =
            rotation * Eigen::Vector3d(((double)imgWidth / 2),
                                       ((double)0 - (double)imgHeight / 2),
                                       f_center);
        // std::cout << objPoint << std::endl;
        cv::Mat map1, map2;
        cv::convertMaps(map, cv::Mat(), map1, map2, CV_16SC2);
        return std::make_pair(map, cv::Mat());
        // return std::make_pair(map1, map2);
    }

    std::pair<cv::Mat, cv::Mat> genOneUndistMap(int _id,
                                                camodocal::CameraPtr p_cam,
                                                Eigen::Quaterniond rotation,
                                                const unsigned &imgWidth,
                                                const unsigned &imgHeight,
                                                const double &f_center) {
        cv::Mat map = cv::Mat(imgHeight, imgWidth, CV_32FC2);
        ROS_DEBUG("Generating map of size (%d,%d)", map.size[0], map.size[1]);
        ROS_DEBUG("Perspective facing (%.2f,%.2f,%.2f)",
                  (rotation * Eigen::Vector3d(0, 0, 1))[0],
                  (rotation * Eigen::Vector3d(0, 0, 1))[1],
                  (rotation * Eigen::Vector3d(0, 0, 1))[2]);
        for (unsigned int x = 0; x < imgWidth; x++)
            for (unsigned int y = 0; y < imgHeight; y++) {
                Eigen::Vector3d objPoint =
                    rotation *
                    Eigen::Vector3d(((double)x - (double)imgWidth / 2),
                                    ((double)y - (double)imgHeight / 2),
                                    f_center);
                Eigen::Vector2d imgPoint;
                p_cam->spaceToPlane(objPoint, imgPoint);

                map.at<cv::Vec2f>(cv::Point(x, y)) =
                    cv::Vec2f(imgPoint.x(), imgPoint.y());
                if (!isnan(imgPoint.x()) && !isnan(imgPoint.y()) &&
                    imgPoint.x() >= 0 && imgPoint.x() <= raw_width &&
                    imgPoint.y() >= 0 && imgPoint.y() <= raw_height) {
                    auto &pt = fisheye2cam_pt.at<cv::Vec2f>(
                        cv::Point(imgPoint.x(), imgPoint.y()));
                    fisheye2cam_id.at<uint8_t>(
                        cv::Point(imgPoint.x(), imgPoint.y())) = _id;
                    pt[0] = x;
                    pt[1] = y;
                }
            }

        ROS_DEBUG("Upper corners: (%.2f, %.2f), (%.2f, %.2f)",
                  map.at<cv::Vec2f>(cv::Point(0, 0))[0],
                  map.at<cv::Vec2f>(cv::Point(0, 0))[1],
                  map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[0],
                  map.at<cv::Vec2f>(cv::Point(imgWidth - 1, 0))[1]);

        Eigen::Vector3d objPoint =
            rotation * Eigen::Vector3d(((double)0 - (double)imgWidth / 2),
                                       ((double)0 - (double)imgHeight / 2),
                                       f_center);

        objPoint =
            rotation * Eigen::Vector3d(((double)imgWidth / 2),
                                       ((double)0 - (double)imgHeight / 2),
                                       f_center);
        cv::Mat map1, map2;
        cv::convertMaps(map, cv::Mat(), map1, map2, CV_16SC2);
        return std::make_pair(map, cv::Mat());
    }
};
}  // namespace D2Common