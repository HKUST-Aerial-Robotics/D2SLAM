#include <d2frontend/loop_cam.h>
#include <camodocal/camera_models/CameraFactory.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include <chrono>
#include <opencv2/core/eigen.hpp>
#include <camodocal/camera_models/Camera.h>
#include <camodocal/camera_models/PinholeCamera.h>
#include <d2frontend/d2featuretracker.h>

using namespace std::chrono;

double TRIANGLE_THRES;

namespace D2FrontEnd {
LoopCam::LoopCam(LoopCamConfig config, ros::NodeHandle &nh) : 
    camera_configuration(config.camera_configuration),
    self_id(config.self_id),
    _config(config)
{
#ifdef USE_TENSORRT
    superpoint_net = new SuperPointTensorRT(config.superpoint_model, config.pca_comp, 
            config.pca_mean, config.width, config.height, config.superpoint_thres, config.superpoint_max_num); 
    if (!config.mobilenetvlad_use_onnx)
        netvlad_net = new MobileNetVLADTensorRT(config.netvlad_model, config.netvlad_width, config.netvlad_height); 
#endif
#ifdef USE_ONNX
    if (config.mobilenetvlad_use_onnx)
        netvlad_onnx = new MobileNetVLADONNX(config.netvlad_model, config.netvlad_width, config.netvlad_height);
#endif

    camodocal::CameraFactory cam_factory;
    for (auto & cam_calib_path : config.camera_config_paths) {
        ROS_INFO("Read camera from %s", cam_calib_path.c_str());
        auto cam = cam_factory.generateCameraFromYamlFile(cam_calib_path);
        if (cam) {
            cams.push_back(cam);
        } else {
            ROS_ERROR("Failed to read camera from %s", cam_calib_path.c_str());
        }
    }
#ifndef USE_TENSORRT
    hfnet_client = nh.serviceClient<HFNetSrv>("/swarm_loop/hfnet");
    superpoint_client = nh.serviceClient<HFNetSrv>("/swarm_loop/superpoint");
    printf("Waiting for deepnet......\n");
    hfnet_client.waitForExistence();
    superpoint_client.waitForExistence();
#endif
    printf("Deepnet ready\n");
    if (_config.OUTPUT_RAW_SUPERPOINT_DESC) {
        fsp.open(params->OUTPUT_PATH+"superpoint.csv", std::fstream::app);
    }
}

void LoopCam::encodeImage(const cv::Mat &_img, VisualImageDesc &_img_desc)
{
    std::vector<int> jpg_params;
    jpg_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    jpg_params.push_back(params->JPG_QUALITY);

    cv::imencode(".jpg", _img, _img_desc.image, jpg_params);
    _img_desc.image_width = _img.cols;
    _img_desc.image_width = _img.rows;
    // std::cout << "IMENCODE Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" << std::endl;
    // std::cout << "JPG SIZE" << _img_desc.image.size() << std::endl;
}

double triangulatePoint(Eigen::Quaterniond q0, Eigen::Vector3d t0, Eigen::Quaterniond q1, Eigen::Vector3d t1,
                      Eigen::Vector2d point0, Eigen::Vector2d point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix3d R0 = q0.toRotationMatrix();
    Eigen::Matrix3d R1 = q1.toRotationMatrix();

    // std::cout << "RO" << R0 << "T0" << t0.transpose() << std::endl;
    // std::cout << "R1" << R1 << "T1" << t1.transpose() << std::endl;

    Eigen::Matrix<double, 3, 4> Pose0;
    Pose0.leftCols<3>() = R0.transpose();
    Pose0.rightCols<1>() = -R0.transpose() * t0;

    Eigen::Matrix<double, 3, 4> Pose1;
    Pose1.leftCols<3>() = R1.transpose();
    Pose1.rightCols<1>() = -R1.transpose() * t1;

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
        design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);

    Eigen::MatrixXd pts(4, 1);
    pts << point_3d.x(), point_3d.y(), point_3d.z(), 1;
    Eigen::MatrixXd errs = design_matrix*pts;
    return errs.norm()/ errs.rows(); 
}

template <typename T>
void reduceVector(std::vector<T> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

cv::Mat drawMatches(std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2, std::vector<cv::DMatch> _matches, const cv::Mat & up, const cv::Mat & down) {
    std::vector<cv::KeyPoint> kps1;
    std::vector<cv::KeyPoint> kps2;

    for (auto pt : pts1) {
        cv::KeyPoint kp;
        kp.pt = pt;
        kps1.push_back(kp);
    }

    for (auto pt : pts2) {
        cv::KeyPoint kp;
        kp.pt = pt;
        kps2.push_back(kp);
    }

    cv::Mat _show;

    cv::drawMatches(up, kps1, down, kps2, _matches, _show);

    return _show;
}

void matchLocalFeatures(std::vector<cv::Point2f> & pts_up, std::vector<cv::Point2f> & pts_down, 
        std::vector<float> & _desc_up, std::vector<float> & _desc_down, 
        std::vector<int> & ids_up, std::vector<int> & ids_down) {
    // printf("matchLocalFeatures %ld %ld: ", pts_up.size(), pts_down.size());
    const cv::Mat desc_up( _desc_up.size()/FEATURE_DESC_SIZE, FEATURE_DESC_SIZE, CV_32F, _desc_up.data());
    const cv::Mat desc_down( _desc_down.size()/FEATURE_DESC_SIZE, FEATURE_DESC_SIZE, CV_32F, _desc_down.data());

    cv::BFMatcher bfmatcher(cv::NORM_L2, true);

    std::vector<cv::DMatch> _matches;
    bfmatcher.match(desc_up, desc_down, _matches);

    std::vector<cv::Point2f> _pts_up, _pts_down;
    std::vector<int> ids;
    for (auto match : _matches) {
        if (match.distance < ACCEPT_SP_MATCH_DISTANCE || true) 
        {
            int now_id = match.queryIdx;
            int old_id = match.trainIdx;
            _pts_up.push_back(pts_up[now_id]);
            _pts_down.push_back(pts_down[old_id]);
            ids_up.push_back(now_id);
            ids_down.push_back(old_id);
        } else {
            std::cout << "Giveup match dis" << match.distance << std::endl;
        }
    }

    std::vector<uint8_t> status;

    pts_up = std::vector<cv::Point2f>(_pts_up);
    pts_down = std::vector<cv::Point2f>(_pts_down);
}

VisualImageDescArray LoopCam::processStereoframe(const StereoFrame & msg, std::vector<cv::Mat> &imgs) {
    VisualImageDescArray visual_array;
    visual_array.stamp = msg.stamp.toSec();
    
    imgs.resize(msg.left_images.size());

    cv::Mat _show, tmp;
    TicToc tt;
    static int t_count = 0;
    static double tt_sum = 0;

    for (unsigned int i = 0; i < msg.left_images.size(); i ++) {
        if (camera_configuration == CameraConfig::PINHOLE_DEPTH) {
            visual_array.images.push_back(generateGrayDepthImageDescriptor(msg, imgs[i], i, tmp));
        } else {
            int camera_index_num = 2;
            if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
                camera_index_num = 5;
            }
            auto _imgs = generateStereoImageDescriptor(msg, imgs[i], i, tmp);
            if (_config.stereo_as_depth_cam) {
                if (_config.right_cam_as_main) {
                    visual_array.images.push_back(_imgs[1]);
                } else {
                    visual_array.images.push_back(_imgs[0]);
                }
            } else {
                visual_array.images.push_back(_imgs[0]);
                visual_array.images.push_back(_imgs[1]);
            }
        }

        if (_show.cols == 0) {
            _show = tmp;
        } else {
            cv::hconcat(_show, tmp, _show);
        }
    }

    tt_sum+= tt.toc();
    t_count+= 1;
    // ROS_INFO("[D2Frontend::LoopCam] KF Count %d loop_cam cost avg %.1fms cur %.1fms", kf_count, tt_sum/t_count, tt.toc());

    visual_array.frame_id = msg.keyframe_id;
    visual_array.pose_drone = msg.pose_drone;
    visual_array.drone_id = self_id;

    if (show && !_show.empty()) {
        char text[100] = {0};
        char PATH[100] = {0};
        sprintf(text, "FEATURES@Drone%d", self_id);
        sprintf(PATH, "loop/features%d.png", kf_count);
        cv::imshow(text, _show);
        cv::imwrite(params->OUTPUT_PATH+PATH, _show);
        cv::waitKey(10);
    }
    kf_count ++;
    visual_array.sync_landmark_ids();
    return visual_array;
}

VisualImageDesc LoopCam::generateGrayDepthImageDescriptor(const StereoFrame & msg, cv::Mat & img, int vcam_id, cv::Mat & _show)
{
    if (vcam_id > msg.left_images.size()) {
        ROS_WARN("Flatten images too few");
        VisualImageDesc ides;
        ides.stamp = msg.stamp.toSec();
        return ides;
    }
    
    VisualImageDesc vframe = extractorImgDescDeepnet(msg.stamp, msg.left_images[vcam_id], msg.left_camera_indices[vcam_id], msg.left_camera_ids[vcam_id], false);

    if (vframe.image_desc.size() == 0)
    {
        ROS_WARN("Failed on deepnet.");
        cv::Mat _img;
        // return ides;
    }

    auto start = high_resolution_clock::now();
    // std::cout << "Downsample and encode Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" << std::endl;

    vframe.extrinsic = msg.left_extrisincs[vcam_id];
    vframe.pose_drone = msg.pose_drone;
    vframe.frame_id = msg.keyframe_id;
    if (params->debug_image || params->ftconfig->enable_lk_optical_flow) {
        vframe.raw_image = msg.left_images[vcam_id];
        vframe.raw_depth_image = msg.depth_images[vcam_id];
    }

    auto image_left = msg.left_images[vcam_id];
    auto pts_up = vframe.landmarks2D();
    std::vector<int> ids_up, ids_down;
    if (vframe.landmarkNum() < _config.ACCEPT_MIN_3D_PTS) {
        return vframe;
    }
    
    Swarm::Pose pose_drone(msg.pose_drone);
    Swarm::Pose pose_cam = pose_drone * Swarm::Pose(msg.left_extrisincs[vcam_id]);

    std::vector<float> desc_new;

    int count_3d = 0;
    for (unsigned int i = 0; i < pts_up.size(); i++)
    {
        cv::Point2f pt_up = pts_up[i];
        if (pt_up.x < 0 || pt_up.x >= params->width || pt_up.y < 0 || pt_up.y >= params->height) {
            continue;
        }
        
        auto dep = msg.depth_images[vcam_id].at<unsigned short>(pt_up)/1000.0;
        if (dep > _config.DEPTH_NEAR_THRES && dep < _config.DEPTH_FAR_THRES) {
            Eigen::Vector3d pt_up3d, pt_down3d;
            cams.at(vcam_id)->liftProjective(Vector2d(pt_up.x, pt_up.y), pt_up3d);

            Eigen::Vector3d pt2d_norm(pt_up3d.x()/pt_up3d.z(), pt_up3d.y()/pt_up3d.z(), 1);
            auto pt3dcam = pt2d_norm*dep;
            Eigen::Vector3d pt3d = pose_cam * pt3dcam;
            // printf("landmark raw depth %f pt3dcam %f %f %f pt2d_norm %f %f distance %f\n", dep, 
            //     pt3dcam.x(), pt3dcam.y(), pt3dcam.z(), pt2d_norm.x(), pt2d_norm.y(), pt3dcam.norm());

            vframe.landmarks[i].pt3d = pt3d;
            vframe.landmarks[i].flag = LandmarkFlag::UNINITIALIZED;
            vframe.landmarks[i].depth = pt3dcam.norm();
            vframe.landmarks[i].depth_mea = true;
            count_3d ++;
        }
    }

    // ROS_INFO("Image 2d kpts: %ld 3d : %d desc size %ld", ides.landmarks_2d.size(), count_3d, ides.feature_descriptor.size());

    if (send_img) {
        encodeImage(image_left, vframe);
    }

    if (show) {
        cv::Mat img_up = image_left;

        img_up.copyTo(img);
        if (!send_img) {
            encodeImage(img_up, vframe);
        }

        cv::cvtColor(img_up, img_up, cv::COLOR_GRAY2BGR);

        for (unsigned int i = 0; i < vframe.landmarkNum(); i++ ) {
            if (vframe.landmarks[i].depth_mea) { 
                auto pt = vframe.landmarks[i].pt2d;
                auto dep = vframe.landmarks[i].depth;
                cv::circle(img_up, pt, 3, cv::Scalar(0, 255, 0), 1);
                char idtext[100] = {};
                sprintf(idtext, "%3.2f", dep);
                cv::putText(img_up, idtext, pt - cv::Point2f(5, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }
        }

        _show = img_up;
        char text[100] = {0};
        sprintf(text, "Frame %d: %ld Features %d/%d", kf_count, msg.keyframe_id, count_3d, pts_up.size());
        cv::putText(_show, text, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
    }
    
    return vframe;
}

std::vector<VisualImageDesc> LoopCam::generateStereoImageDescriptor(const StereoFrame & msg, cv::Mat & img, int vcam_id, cv::Mat & _show)
{
    //This function currently only support pinhole-like stereo camera.
    auto vframe0 = extractorImgDescDeepnet(msg.stamp, msg.left_images[vcam_id], msg.left_camera_indices[vcam_id], 
        msg.left_camera_ids[vcam_id], _config.right_cam_as_main);
    auto vframe1 = extractorImgDescDeepnet(msg.stamp, msg.right_images[vcam_id], msg.right_camera_indices[vcam_id], 
        msg.right_camera_ids[vcam_id], !_config.right_cam_as_main);

    if (vframe0.image_desc.size() == 0 && vframe1.image_desc.size() == 0)
    {
        ROS_WARN("Failed on deepnet;");
        // cv::Mat _img;
        // return ides;
    }

    auto start = high_resolution_clock::now();
    // std::cout << "Downsample and encode Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" << std::endl;

    vframe0.extrinsic = msg.left_extrisincs[vcam_id];
    vframe0.pose_drone = msg.pose_drone;
    vframe0.frame_id = msg.keyframe_id;

    vframe1.extrinsic = msg.right_extrisincs[vcam_id];
    vframe1.pose_drone = msg.pose_drone;
    vframe1.frame_id = msg.keyframe_id;

    auto image_left = msg.left_images[vcam_id];
    auto image_right = msg.right_images[vcam_id];

    auto pts_up = vframe0.landmarks2D();
    auto pts_down = vframe1.landmarks2D();
    std::vector<int> ids_up, ids_down;

    int count_3d = 0;
    if (_config.stereo_as_depth_cam) {
        if (vframe0.landmarkNum() > _config.ACCEPT_MIN_3D_PTS) {
            matchLocalFeatures(pts_up, pts_down, vframe0.landmark_descriptor, vframe1.landmark_descriptor, ids_up, ids_down);
        }
        Swarm::Pose pose_drone(msg.pose_drone);
        Swarm::Pose pose_up = pose_drone * Swarm::Pose(msg.left_extrisincs[vcam_id]);
        Swarm::Pose pose_down = pose_drone * Swarm::Pose(msg.right_extrisincs[vcam_id]);
        for (unsigned int i = 0; i < pts_up.size(); i++) {
            auto pt_up = pts_up[i];
            auto pt_down = pts_down[i];

            Eigen::Vector3d pt_up3d, pt_down3d;
            //TODO: This may not work for stereo fisheye. Pending to update.
            cams.at(msg.left_camera_indices[vcam_id])->liftProjective(Eigen::Vector2d(pt_up.x, pt_up.y), pt_up3d);
            cams.at(msg.right_camera_indices[vcam_id])->liftProjective(Eigen::Vector2d(pt_down.x, pt_down.y), pt_down3d);

            Eigen::Vector2d pt_up_norm(pt_up3d.x()/pt_up3d.z(), pt_up3d.y()/pt_up3d.z());
            Eigen::Vector2d pt_down_norm(pt_down3d.x()/pt_down3d.z(), pt_down3d.y()/pt_down3d.z());

            Eigen::Vector3d point_3d;
            double err = triangulatePoint(pose_up.att(), pose_up.pos(), pose_down.att(), pose_down.pos(),
                            pt_up_norm, pt_down_norm, point_3d);

            auto pt_cam = pose_up.att().inverse() * (point_3d - pose_up.pos());

            if (err > TRIANGLE_THRES) {
                continue;
            }

            int idx = ids_up[i];
            int idx_down = ids_down[i];
            vframe0.landmarks[idx].pt3d = point_3d;
            vframe0.landmarks[idx].depth_mea = true;
            vframe0.landmarks[idx].depth = pt_cam.norm();
            vframe0.landmarks[idx].flag = LandmarkFlag::UNINITIALIZED; 

            auto pt_cam2 = pose_down.att().inverse() * (point_3d - pose_down.pos());
            vframe1.landmarks[idx_down].pt3d = point_3d;
            vframe0.landmarks[idx].depth_mea = true;
            vframe0.landmarks[idx].depth = pt_cam2.norm();
            vframe1.landmarks[idx_down].flag = LandmarkFlag::UNINITIALIZED;
            count_3d ++;
        }
    }

    if (send_img) {
        if (_config.right_cam_as_main) {
            encodeImage(image_right, vframe1);
        } else {
            encodeImage(image_left, vframe0);
        }
    }

    if (show) {
        cv::Mat img_up = image_left;
        cv::Mat img_down = image_right;

        img_up.copyTo(img);
        if (!send_img) {
            encodeImage(img_up, vframe0);
            encodeImage(img_down, vframe1);
        }

        cv::cvtColor(img_up, img_up, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img_down, img_down, cv::COLOR_GRAY2BGR);

        for (auto pt : pts_down)
        {
            cv::circle(img_down, pt, 1, cv::Scalar(255, 0, 0), -1);
        }

        for (auto _pt : vframe0.landmarks2D()) {
            cv::Point2f pt(_pt.x, _pt.y);
            cv::circle(img_up, pt, 3, cv::Scalar(0, 0, 255), 1);
        }

        cv::vconcat(img_up, img_down, _show);
        if (_config.stereo_as_depth_cam) {
            for (unsigned int i = 0; i < pts_up.size(); i++)
            {
                int idx = ids_up[i];
                if (vframe0.landmarks[idx].flag) {
                    char title[100] = {0};
                    auto pt = pts_up[i];
                    cv::circle(_show, pt, 3, cv::Scalar(0, 255, 0), 1);
                    cv::arrowedLine(_show, pts_up[i], pts_down[i], cv::Scalar(255, 255, 0), 1);
                }
            }
        }

        char text[100] = {0};
        sprintf(text, "Frame %ld Features %d/%d", msg.keyframe_id, count_3d, pts_up.size());
        cv::putText(_show, text, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
    }

    if (params->debug_image) {
        vframe1.raw_image = msg.right_images[vcam_id];
        vframe0.raw_image = msg.left_images[vcam_id];
    }
    std::vector<VisualImageDesc> ret{vframe0, vframe1};
    return ret;
}

VisualImageDesc LoopCam::extractorImgDescDeepnet(ros::Time stamp, cv::Mat img, int camera_index, 
        int camera_id, bool superpoint_mode)
{
    auto start = high_resolution_clock::now();

    VisualImageDesc vframe;
    vframe.stamp = stamp.toSec();
    vframe.camera_index = camera_index;
    vframe.camera_id = camera_id;
    vframe.drone_id = self_id;

    if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
        cv::Mat roi = img(cv::Rect(0, img.rows*3/4, img.cols, img.rows/4));
        roi.setTo(cv::Scalar(0, 0, 0));
    }
    std::vector<cv::Point2f> landmarks_2d;
    if (_config.superpoint_max_num > 0) {
        //We only inference when superpoint max num > 0
        //otherwise, d2vins only uses LK optical flow feature.
#ifdef USE_TENSORRT
        superpoint_net->inference(img, landmarks_2d, vframe.landmark_descriptor);
#endif
    }

    if (!superpoint_mode) {
        if (_config.mobilenetvlad_use_onnx) {
#ifdef USE_ONNX
            vframe.image_desc = netvlad_onnx->inference(img);
#endif
        } else {
#ifdef USE_TENSORRT
            vframe.image_desc = netvlad_net->inference(img);
#endif
        }
    }

    for (unsigned int i = 0; i < landmarks_2d.size(); i++)
    {
        auto pt_up = landmarks_2d[i];
        Eigen::Vector3d pt_up3d;
        cams.at(camera_index)->liftProjective(Eigen::Vector2d(pt_up.x, pt_up.y), pt_up3d);
        LandmarkPerFrame lm;
        lm.pt2d = pt_up;
        pt_up3d.normalize();
        lm.pt3d_norm = pt_up3d;
        lm.camera_index = camera_index;
        lm.camera_id = camera_id;
        lm.stamp = vframe.stamp;
        lm.color = extractColor(img, pt_up);
        vframe.landmarks.emplace_back(lm);
        if (_config.OUTPUT_RAW_SUPERPOINT_DESC) {
            for (int j = 0; j < FEATURE_DESC_SIZE; j ++) {
                fsp << vframe.landmark_descriptor[i*FEATURE_DESC_SIZE + j] << " ";
            }
            fsp << std::endl;
        }
    } 

    return vframe;
#ifndef USE_TENSORRT
    HFNetSrv hfnet_srv;
    hfnet_srv.request.image = msg;
    if (superpoint_mode) {
        if (superpoint_client.call(hfnet_srv))
        {
            auto &local_kpts = hfnet_srv.response.keypoints;
            auto &local_descriptors = hfnet_srv.response.local_descriptors;
            if (local_kpts.size() > 0)
            {
                // ROS_INFO("Received response from server desc.size %ld", desc.size());
                // ROSPoints2LCM(local_kpts, img_des.landmarks_2d);
                vframe.feature_descriptor = local_descriptors;
                vframe.landmarks_flag.resize(img_des.landmarkNum());
                std::fill(vframe.landmarks_flag.begin(),vframe.landmarks_flag.begin()+vframe.landmarkNum(),0);  
                return vframe;
            }
        }
    } else {
        if (hfnet_client.call(hfnet_srv))
        {
            auto &desc = hfnet_srv.response.global_desc;
            auto &local_kpts = hfnet_srv.response.keypoints;
            auto &local_descriptors = hfnet_srv.response.local_descriptors;
            if (desc.size() > 0)
            {
                // ROS_INFO("Received response from server desc.size %ld", desc.size());
                vframe.image_desc = desc;
                ROSPoints2LCM(local_kpts, img_des.landmarks_2d);
                img_des.landmark_num = local_kpts.size();
                vframe.feature_descriptor = local_descriptors;
                vframe.landmarks_flag.resize(vframe.landmarkNum());
                std::fill(vframe.landmarks_flag.begin(),vframe.landmarks_flag.begin()+vframe.landmark_num,0);  
                return vframe;
            }
        }
    }
#endif
}
}