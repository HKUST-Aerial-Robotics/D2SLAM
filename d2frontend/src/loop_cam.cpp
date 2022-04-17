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

using namespace std::chrono;

double TRIANGLE_THRES;

namespace D2FrontEnd {
LoopCam::LoopCam(LoopCamConfig config, ros::NodeHandle &nh) : 
    camera_configuration(config.camera_configuration),
    self_id(config.self_id),
#ifdef USE_TENSORRT
    superpoint_net(config.superpoint_model, config.pca_comp, 
        config.pca_mean, config.width, config.height, 
        config.superpoint_thres, 
        config.superpoint_max_num), 
    netvlad_net(config.netvlad_model, config.width, config.height), 
#endif
    _config(config)
{
    camodocal::CameraFactory cam_factory;
    ROS_INFO("Read camera from %s", config.camera_config_path.c_str());
    cam = cam_factory.generateCameraFromYamlFile(config.camera_config_path);

#ifndef USE_TENSORRT
    hfnet_client = nh.serviceClient<HFNetSrv>("/swarm_loop/hfnet");
    superpoint_client = nh.serviceClient<HFNetSrv>("/swarm_loop/superpoint");
    printf("Waiting for deepnet......\n");
    hfnet_client.waitForExistence();
    superpoint_client.waitForExistence();
#endif
    camodocal::PinholeCamera* _cam = (camodocal::PinholeCamera*)cam.get();
    Eigen::Matrix3d _cameraMatrix;
    _cameraMatrix << _cam->getParameters().fx(), 0, _cam->getParameters().cx(),
                    0, _cam->getParameters().fy(), _cam->getParameters().cy(), 0, 0, 1;
    cv::eigen2cv(_cameraMatrix, cameraMatrix);
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

    printf("%ld matches...\n", _matches.size());

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
            visual_array.images[i].camera_id = i;
        } else {
            visual_array.images.push_back(generateStereoImageDescriptor(msg, imgs[i], i, tmp));
            visual_array.images[i].camera_id = i;
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

VisualImageDesc LoopCam::generateGrayDepthImageDescriptor(const StereoFrame & msg, cv::Mat & img, const int & vcam_id, cv::Mat & _show)
{
    if (vcam_id > msg.left_images.size()) {
        ROS_WARN("Flatten images too few");
        VisualImageDesc ides;
        ides.stamp = msg.stamp.toSec();
        return ides;
    }
    
    VisualImageDesc vframe = extractorImgDescDeepnet(msg.stamp, msg.left_images[vcam_id], _config.LOWER_CAM_AS_MAIN);

    if (vframe.image_desc.size() == 0)
    {
        ROS_WARN("Failed on deepnet.");
        cv::Mat _img;
        // return ides;
    }

    auto start = high_resolution_clock::now();
    // std::cout << "Downsample and encode Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" << std::endl;

    vframe.stamp = msg.stamp.toSec();
    vframe.drone_id = self_id; // -1 is self drone;
    vframe.extrinsic = msg.left_extrisincs[vcam_id];
    vframe.pose_drone = msg.pose_drone;
    vframe.frame_id = msg.keyframe_id;
    if (params->debug_image) {
        vframe.raw_image = msg.left_images[vcam_id];
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
        if (pt_up.x < 0 || pt_up.x > 640 || pt_up.y < 0 || pt_up.y > 480) {
            continue;
        }
        
        auto dep = msg.depth_images[vcam_id].at<unsigned short>(pt_up)/1000.0;
        if (dep > _config.DEPTH_NEAR_THRES && dep < _config.DEPTH_FAR_THRES) {
            Eigen::Vector3d pt_up3d, pt_down3d;
            cam->liftProjective(Vector2d(pt_up.x, pt_up.y), pt_up3d);

            Eigen::Vector3d _pt3d(pt_up3d.x()/pt_up3d.z(), pt_up3d.y()/pt_up3d.z(), 1);
            _pt3d = pose_cam * (_pt3d*dep);

            vframe.landmarks[i].pt3d = _pt3d;
            vframe.landmarks[i].flag = 1;
            vframe.landmarks[i].depth = dep;
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
            if (vframe.landmarks[i].flag) { 
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

VisualImageDesc LoopCam::generateStereoImageDescriptor(const StereoFrame & msg, cv::Mat & img, const int & vcam_id, cv::Mat & _show)
{
    if (vcam_id > msg.left_images.size()) {
        ROS_WARN("Flatten images too few");
        VisualImageDesc ides;
        return ides;
    }
    
    auto vframe0 = extractorImgDescDeepnet(msg.stamp, msg.left_images[vcam_id], _config.LOWER_CAM_AS_MAIN);
    auto vframe1 = extractorImgDescDeepnet(msg.stamp, msg.right_images[vcam_id], !_config.LOWER_CAM_AS_MAIN);

    if (vframe0.image_desc.size() == 0 && vframe1.image_desc.size() == 0)
    {
        ROS_WARN("Failed on deepnet;");
        // cv::Mat _img;
        // return ides;
    }

    auto start = high_resolution_clock::now();
    // std::cout << "Downsample and encode Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" << std::endl;

    vframe0.stamp = msg.stamp.toSec();
    vframe0.drone_id = self_id; // -1 is self drone;
    vframe0.extrinsic = msg.left_extrisincs[vcam_id];
    vframe0.pose_drone = msg.pose_drone;
    vframe0.frame_id = msg.keyframe_id;

    vframe1.stamp = msg.stamp.toSec();
    vframe1.drone_id = self_id; // -1 is self drone;
    vframe1.extrinsic = msg.right_extrisincs[vcam_id];
    vframe1.pose_drone = msg.pose_drone;
    vframe1.frame_id = msg.keyframe_id;

    auto image_left = msg.left_images[vcam_id];
    auto image_right = msg.right_images[vcam_id];

    auto pts_up = vframe0.landmarks2D();
    auto pts_down = vframe1.landmarks2D();
    std::vector<int> ids_up, ids_down;

    if (vframe0.landmarkNum() > _config.ACCEPT_MIN_3D_PTS) {
        matchLocalFeatures(pts_up, pts_down, vframe0.landmark_descriptor, vframe1.landmark_descriptor, ids_up, ids_down);
    } else {
        return vframe0;
    }
    
    // ides.landmarks_2d.clear();
    // ides.landmarks_2d_norm.clear();
    // ides.landmarks_3d.clear();
    
    Swarm::Pose pose_drone(msg.pose_drone);
    Swarm::Pose pose_up = pose_drone * Swarm::Pose(msg.left_extrisincs[vcam_id]);
    Swarm::Pose pose_down = pose_drone * Swarm::Pose(msg.right_extrisincs[vcam_id]);

    std::vector<float> desc_new;

    int count_3d = 0;

    for (unsigned int i = 0; i < pts_up.size(); i++)
    {
        auto pt_up = pts_up[i];
        auto pt_down = pts_down[i];

        Eigen::Vector3d pt_up3d, pt_down3d;
        cam->liftProjective(Eigen::Vector2d(pt_up.x, pt_up.y), pt_up3d);
        cam->liftProjective(Eigen::Vector2d(pt_down.x, pt_down.y), pt_down3d);

        Eigen::Vector2d pt_up_norm(pt_up3d.x()/pt_up3d.z(), pt_up3d.y()/pt_up3d.z());
        Eigen::Vector2d pt_down_norm(pt_down3d.x()/pt_down3d.z(), pt_down3d.y()/pt_down3d.z());

        Eigen::Vector3d point_3d;
        double err = triangulatePoint(pose_up.att(), pose_up.pos(), pose_down.att(), pose_down.pos(),
                        pt_up_norm, pt_down_norm, point_3d);

        auto pt_cam = pose_up.att().inverse() * (point_3d - pose_up.pos());

        if (err > TRIANGLE_THRES || pt_cam.z() < 0) {
            continue;
        }

        int idx = ids_up[i];
        int idx_down = ids_down[i];
        // ides.landmarks_2d.push_back(pt2d);
        // ides.landmarks_2d_norm.push_back(pt2d_norm);
        vframe0.landmarks[idx].pt3d = point_3d;
        vframe0.landmarks[idx].flag = 1; 
        //TODO:Set depth!!!

        vframe1.landmarks[idx_down].pt3d = point_3d;
        vframe1.landmarks[idx_down].flag = 1;
        count_3d ++;
        assert("Set depth before use" && false);
        // std::cout << "Insert" << FEATURE_DESC_SIZE * ids[i] << "to" << FEATURE_DESC_SIZE * (ids[i] + 1)  << std::endl;

        // desc_new.insert(desc_new.end(), ides.feature_descriptor.begin() + FEATURE_DESC_SIZE * ids[i], ides.feature_descriptor.begin() + FEATURE_DESC_SIZE * (ids[i] + 1) );

        // std::cout << "PT UP" << pt_up << "PT DOWN" << pt_down << std::endl;

        // std::cout << "PT UP NORM" << pt_up_norm.transpose() << "PT DOWN NORM" << pt_down_norm.transpose() << std::endl;

    }

    // ROS_INFO("Image 2d kpts: %ld 3d : %d", ides.landmarks_2d.size(), count_3d);

    // ides.feature_descriptor.clear();
    // ides.feature_descriptor = std::vector<float>(desc_new);
    // ides.feature_descriptor_size = ides.feature_descriptor.size();
    // ides.landmark_num = ides.landmarks_2d.size();

    if (send_img) {
        if (_config.LOWER_CAM_AS_MAIN) {
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
        for (unsigned int i = 0; i < pts_up.size(); i++)
        {
            int idx = ids_up[i];
            if (vframe0.landmarks[idx].flag) {
                char title[100] = {0};
                auto pt = pts_up[i];
                auto point3d = vframe0.landmarks[idx].pt3d;
                auto pt_cam = pose_up.att().inverse() * (point3d - pose_up.pos());
                cv::circle(_show, pt, 3, cv::Scalar(0, 255, 0), 1);
                cv::arrowedLine(_show, pts_up[i], pts_down[i], cv::Scalar(255, 255, 0), 1);

                // sprintf(title, "[%3.1f,%3.1f,%3.1f]", pt_cam.x(), pt_cam.y(), pt_cam.z());
                // cv::putText(_show, title, pt + cv::Point2f(0, 5), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 0), 1);
            }
        }

        char text[100] = {0};
        sprintf(text, "Frame %ld Features %d/%d", msg.keyframe_id, count_3d, pts_up.size());
        cv::putText(_show, text, cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.5);
    }

    if (_config.LOWER_CAM_AS_MAIN) {
        if (params->debug_image) {
            vframe1.raw_image = msg.left_images[vcam_id];
        }
        return vframe1;
    } else {
        if (params->debug_image) {
            vframe0.raw_image = msg.left_images[vcam_id];
        }
        return vframe0;
    }
}

VisualImageDesc LoopCam::extractorImgDescDeepnet(ros::Time stamp, cv::Mat img, bool superpoint_mode)
{
    auto start = high_resolution_clock::now();

    VisualImageDesc vframe;
    vframe.stamp = stamp.toSec();

    if (camera_configuration == CameraConfig::STEREO_FISHEYE) {
        cv::Mat roi = img(cv::Rect(0, img.rows*3/4, img.cols, img.rows/4));
        roi.setTo(cv::Scalar(0, 0, 0));
    }
#ifdef USE_TENSORRT
    std::vector<cv::Point2f> landmarks_2d;
    superpoint_net.inference(img, landmarks_2d, vframe.landmark_descriptor);

    if (!superpoint_mode) {
        vframe.image_desc = netvlad_net.inference(img);
    }

    for (unsigned int i = 0; i < landmarks_2d.size(); i++)
    {
        auto pt_up = landmarks_2d[i];
        Eigen::Vector3d pt_up3d;
        cam->liftProjective(Eigen::Vector2d(pt_up.x, pt_up.y), pt_up3d);
        Eigen::Vector2d pt_up_norm(pt_up3d.x()/pt_up3d.z(), pt_up3d.y()/pt_up3d.z());
        LandmarkPerFrame lm;
        lm.pt2d = pt_up;
        lm.pt2d_norm = pt_up_norm;
        vframe.landmarks.emplace_back(lm);

        if (_config.OUTPUT_RAW_SUPERPOINT_DESC) {
            for (int j = 0; j < FEATURE_DESC_SIZE; j ++) {
                fsp << vframe.landmark_descriptor[i*FEATURE_DESC_SIZE + j] << " ";
            }
            fsp << std::endl;
        }
    } 

    return vframe;
#else
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