#!/usr/bin/env python3
from test_depth_estimation import *
from quad_cam_split import split_image
from cv_bridge import CvBridge
import cv2 as cv
import rospy
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import time

bridge = CvBridge()
FIELDS = [PointField('x', 0, PointField.FLOAT32, 1),
    PointField('y', 4, PointField.FLOAT32, 1),
    PointField('z', 8, PointField.FLOAT32, 1),
    PointField('b', 12, PointField.FLOAT32, 1),
    PointField('g', 16, PointField.FLOAT32, 1),
    PointField('r', 20, PointField.FLOAT32, 1)]

class DepthGenerateNode:
    def __init__(self, fisheye_configs, stereo_paths, photometric_path, fov=190, width=600, height=300):
        self.gens, self.is_rgb = loadConfig(fisheye_configs, stereo_paths, fov=fov, width=width, height=height, hitnet=True)
        self.photometric = cv.imread(photometric_path, cv.IMREAD_GRAYSCALE)/255.0
        self.pcl_pub = rospy.Publisher("/depth_estimation/point_cloud_py", PointCloud2, queue_size=1)
        self.max_z = 100
        self.min_z = 0.3
        self.step = 5 #Generate cloud per 3 frames
        self.count = 0
        self.enable_texture = True

    def callback(self, img_msg):
        if self.count % self.step != 0:
            self.count += 1
            return
        s0 = time.time()
        if img_msg._type == "sensor_msgs/Image":
            img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        else:
            img = bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        imgs = split_image(img)
        photometric_calibed = []
        photometric = self.photometric
        for img in imgs:
            #Apply inverse of photometric calibration
            if photometric is not None:
                #Convert to grayscale
                if len(img.shape) == 3 and not self.is_rgb:
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                calibed = calib_photometric(img, photometric, self.is_rgb)
                # calibed= img
                photometric_calibed.append(calibed)
            else:
                photometric_calibed.append(img)
        pcl, texture = None, None
        s = time.time()
        for gen in self.gens[1:]:
            if args.verbose:
                _pcl, _texture = test_depth_gen(gen, photometric_calibed, imgs, detailed=args.verbose)
            else:
                if gen.is_rgb:
                    _pcl, _texture = gen.genPointCloud(photometric_calibed[gen.cam_idx_a], 
                            imgs[gen.cam_idx_b], img_raw=imgs[gen.cam_idx_a], 
                            min_z=self.min_z, max_z=self.max_z, enable_texture=self.enable_texture)
                else:
                    _pcl, _texture = gen.genPointCloud(photometric_calibed[gen.cam_idx_a], 
                            photometric_calibed[gen.cam_idx_b], img_raw=imgs[gen.cam_idx_a], 
                            min_z=self.min_z, max_z=self.max_z, enable_texture=self.enable_texture)
            if pcl is None:
                pcl = _pcl
                texture = _texture
            else:
                pcl = np.concatenate((pcl, _pcl), axis=0)
                if self.enable_texture:
                    texture = np.concatenate((texture, _texture), axis=0)
        tcloud = (time.time() - s)
        header = img_msg.header
        header.frame_id = "imu"
        if self.enable_texture:
            colored_pcl = np.c_[pcl, texture]
            msg = pc2.create_cloud(header, FIELDS, colored_pcl)
        else:
            msg = pc2.create_cloud_xyz32(header, pcl)
        self.pcl_pub.publish(msg)
        self.count += 1
        print(f"Total time {(time.time() - s0)*1000:.1f}ms cloud gen time: {tcloud*1000:.1f}ms")


if __name__ == "__main__":
    rospy.init_node('depth_generate_node')
    #Register node
    parser = argparse.ArgumentParser(description='Fisheye undist')
    parser.add_argument("-f","--fov", type=float, default=180, help="hoizon fov of fisheye")
    parser.add_argument("-c","--config", type=str, default="/home/dji/d2slam_ws/src/D2SLAM/config/quadcam/quad_cam_calib-camchain-imucam.yaml", help="config file path")
    parser.add_argument("-p","--photometric", type=str, help="photometric calibration image", default="/home/dji/d2slam_ws/src/D2SLAM/config/quadcam/mask.png")
    parser.add_argument("-v","--verbose", action='store_true', help="show image")
    parser.add_argument("-w","--width", type=int, default=320, help="width of pinhole")
    parser.add_argument("--height", type=int, default=240, help="width of pinhole")
    args = parser.parse_args()
    stereo_paths = ["/home/dji/d2slam_ws/src/D2SLAM/config/quadcam/stereo_calib_1_0.yaml",
                "/home/dji/d2slam_ws/src/D2SLAM/config/quadcam/stereo_calib_2_1.yaml",
                "/home/dji/d2slam_ws/src/D2SLAM/config/quadcam/stereo_calib_3_2.yaml",
                "/home/dji/d2slam_ws/src/D2SLAM/config/quadcam/stereo_calib_0_3.yaml"]
    node = DepthGenerateNode(args.config, stereo_paths, args.photometric, fov=args.fov, width=args.width, height=args.height)
    #Subscribe to image using ImageTransport
    sub_comp = rospy.Subscriber("/arducam/image/compressed", CompressedImage, node.callback)
    # sub_raw = rospy.Subscriber("/arducam/image", Image, node.callback)
    print("depth_generate_node started")
    rospy.spin()
