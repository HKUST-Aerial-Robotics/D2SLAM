#!/usr/bin/env python3
import rosbag
from os.path import exists
from cv_bridge import CvBridge
import cv2 as cv
import tqdm
import numpy as np

K = np.array([[1162.5434300524314, 0, 660.6393183718625],
            [0, 1161.839362615319,  386.1663300322095],
            [0, 0, 1]])
Knew = np.array([[1000, 0, 640],
            [0, 1000,  400],
            [0, 0, 1]], dtype=np.float32)
D = np.array([-0.17703529535292872, 0.7517933338735744, -0.0008911425891703079, 2.1653595535258756e-05])
xi = np.array(2.2176903753419963)

map1, map2 = cv.omnidir.initUndistortRectifyMap(K, D, xi, np.eye(3), Knew, (1280, 800), 
        cv.CV_32FC1, cv.omnidir.RECTIFY_CYLINDRICAL) 

def show_undist(img, K, D, xi):
    cv.imshow("raw", img)
    img = cv.omnidir.undistortImage(img, K, D, xi, cv.omnidir.RECTIFY_CYLINDRICAL, Knew=Knew)
    cv.imshow("undistorted", img)
    cv.waitKey(1)

def show_undist_by_map(img, map1, map2):
    cv.imwrite("/home/xuhao/output/quadvins-output/raw.jpg", img)
    cv.imshow("raw", img)
    img = cv.remap(img, map1, map2, cv.INTER_LINEAR)
    cv.imshow("undistorted", img)
    cv.waitKey(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split quadcam images')
    parser.add_argument("-i","--input", type=str, help="input bag file")
    parser.add_argument("-t","--topic", type=str, default="/arducam/image_0/compressed", help="input bag file")
    parser.add_argument('-s', '--step', type=int, nargs="?", help="step for images, default 1", default=1)
    args = parser.parse_args()
    bag = rosbag.Bag(args.input)
    num_imgs = bag.get_message_count("/arducam/image/compressed") + bag.get_message_count("/arducam/image")
    print("Total number of images:", num_imgs)
    bridge = CvBridge()
    pbar = tqdm.tqdm(total=num_imgs/args.step, colour="green")
    c = 0
    for topic, msg, t in bag.read_messages():
        if topic == args.topic:
            c += 1
            if c % args.step != 0:
                continue
            if msg._type == "sensor_msgs/Image":
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            show_undist_by_map(img, map1, map2)