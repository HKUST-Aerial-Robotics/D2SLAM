#!/usr/bin/env python3
from stereo_gen import *
from fisheye_undist import *
import cv2 as cv
import numpy as np
import argparse
import rosbag
import tqdm
from cv_bridge import CvBridge
from quad_cam_split import split_image

def loadConfig(config_file):
    import yaml
    undists = []
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for v in config:
            print(v)
            intrinsics = config[v]['intrinsics']
            distortion_coeffs = config[v]['distortion_coeffs']
            xi = intrinsics[0]
            gamma1 = intrinsics[1]
            gamma2 = intrinsics[2]
            u0 = intrinsics[3]
            v0 = intrinsics[4]
            K = np.array([[gamma1 , 0, u0],
                    [0, gamma2, v0],
                    [0, 0, 1]])
            D = np.array(distortion_coeffs)
            undist = FisheyeUndist(K, D, xi, fov=args.fov)
            undists.append(undist)
    gens = [StereoGen(undists[1], undists[0], np.eye(3), np.zeros(3)),
            StereoGen(undists[2], undists[1], np.eye(3), np.zeros(3)),
            StereoGen(undists[3], undists[1], np.eye(3), np.zeros(3)),
            StereoGen(undists[3], undists[2], np.eye(3), np.zeros(3))]
    return gens

class FeatureImage:
    def __init__(self, pts2d, pts3d, idx):
        self.pts2d = pts2d
        self.pts3d = pts3d
        self.idx = idx

def parseMarkerFromBag(bag):
    feature_imgs = [] #num_frame,num_cam
    bag = rosbag.Bag(args.input)
    num_imgs = bag.get_message_count("/arducam/image/compressed") + bag.get_message_count("/arducam/image") \
          + bag.get_message_count("/oak_ffc_4p/assemble_image/compressed") + bag.get_message_count("/oak_ffc_4p/assemble_image")
    print("Total number of images:", num_imgs)
    bridge = CvBridge()
    pbar = tqdm.tqdm(total=num_imgs/args.step, colour="green")
    cv.namedWindow("stereo", cv.WINDOW_NORMAL|cv.WINDOW_GUI_EXPANDED)
    count = 0
    for topic, msg, t in bag.read_messages():
        try:
            if topic == "/arducam/image/compressed" or topic == "/arducam/image/raw" or topic == "oak_ffc_4p/assemble_image/compressed" or topic == "/oak_ffc_4p/assemble_image":
                if count % args.step != 0:
                    count += 1
                    continue
                if msg._type == "sensor_msgs/Image":
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                else:
                    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                imgs = split_image(img)
                pbar.update(1)
                c = cv.waitKey(0)
                if c == ord('q'):
                    break
                count += 1
        except KeyboardInterrupt:
            break