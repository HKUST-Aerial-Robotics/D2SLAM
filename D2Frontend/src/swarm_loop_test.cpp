#include <swarm_loop/loop_cam.h>
#include <opencv2/opencv.hpp>
#include <swarm_msgs/swarm_lcm_converter.hpp>
#include <swarm_loop/loop_detector.h>
#include <chrono> 

using namespace std::chrono; 

#define DEBUG_IMAGE

LoopDetector ld("/home/xuhao/swarm_ws/src/swarm_localization/support_files/ORBvoc.txt");

std::string camera_config_path = 
        "/home/xuhao/swarm_ws/src/VINS-Fusion-gpu/config/vi_car/cam0_mei.yaml";
std::string BRIEF_PATTHER_FILE = "/home/xuhao/swarm_ws/src/VINS-Fusion-gpu/support_files/brief_pattern.yml";

std::string camera_topic = "/cam0/image_raw";
std::string viokeyframe_topic = "/vins_estimator/viokeyframe";

LoopCam cam(camera_config_path, BRIEF_PATTHER_FILE);

int test_loop_cam(ros::NodeHandle & nh) {
    std::string test_img_file = "/home/xuhao/image001.png";
    ros::Publisher img_des_pub = nh.advertise<swarm_msgs::ImageDescriptor>("/swarm_loop/new_image_des", 1);
    cv::Mat img = cv::imread(test_img_file.c_str(), cv::IMREAD_GRAYSCALE);
    auto des =  toROSMsg(cam.feature_detect(img));
    ros::Rate r(10);
    while (true) {
        ROS_INFO("Publishing the descriptors....");
        img_des_pub.publish(des);
        ros::spinOnce();
        r.sleep();
    }
}


void image_callback_0(const sensor_msgs::ImageConstPtr& msg) {
    static int c = 0;
    if (c % 10 == 0) {
        cv_bridge::CvImageConstPtr ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        auto img_des = cam.feature_detect(ptr->image);
        ld.on_image_recv(img_des, ptr->image);
    }
    c++;
}


void image_callback_1(const sensor_msgs::ImageConstPtr& msg) {
    cam.on_camera_message(msg);
}

void VIOKF_callback(const vins::VIOKeyframe & viokf) {
    auto start = high_resolution_clock::now();
    auto ret = cam.on_keyframe_message(viokf);
    std::cout << "Cam Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" << std::endl;
    //Check ides vaild
#ifdef DEBUG_IMAGE
    ld.on_image_recv(ret.first, ret.second);
#else
    ld.on_image_recv(ret.first);
#endif
    std::cout << "Cam+LD Cost " << duration_cast<microseconds>(high_resolution_clock::now() - start).count()/1000.0 << "ms" <<  std::endl;
}

int test_loop_detector(ros::NodeHandle & nh) {
    auto sb = nh.subscribe(camera_topic, 1000, image_callback_0);
    ros::spin();
}

int test_single_loop(ros::NodeHandle & nh) {
    auto sb = nh.subscribe(camera_topic, 1000, image_callback_1);
    auto sb2 = nh.subscribe(viokeyframe_topic, 1000, VIOKF_callback);
    ros::spin();
}


int main(int argc, char **argv) {
    ROS_INFO("SWARM_LOOP INIT");
    srand(time(NULL));

    ros::init(argc, argv, "swarm_loop_test");
    ros::NodeHandle nh("swarm_loop_test");

    //test_loop_detector(nh);
    test_single_loop(nh);
    
    return 0;
}