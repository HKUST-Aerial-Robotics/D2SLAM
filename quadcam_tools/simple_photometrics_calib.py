#!/usr/bin/env python3
from unittest.result import failfast
from stereo_gen import *
from fisheye_undist import *
import cv2 as cv
import numpy as np
import argparse
import rosbag
import tqdm
from cv_bridge import CvBridge
from quad_cam_split import split_image
import matplotlib.pyplot as plt

def processImg(img_input):
    #Mirror image and average with input
    img = cv.flip(img_input, 1)
    img = (img + img_input) / 2
    #Mirror again
    img2 = cv.flip(img, 0)
    img2 = (img2 + img) / 2
    return img2

def findPhotometric(cul_img):
    #Average in polar coordinates
    w_2 = cul_img.shape[1] // 2
    h_2 = cul_img.shape[0] // 2
    w = cul_img.shape[1]
    h = cul_img.shape[0]
    cul_line = np.zeros((w_2, 1)) #Half the width of the image
    count_line = np.zeros((w_2, 1)) #Half the width of the image
    pixels = w_2 * 2
    for theta in range(0, 360, 1):
        for r in np.linspace(0, w_2, pixels):
            x = int(r * np.cos(theta))
            y = int(r * np.sin(theta))
            if x + w_2 < 0 or x + w_2 >= w or y + h_2 >= h or y + h_2 < 20 or int(r) >= w_2: #Ignore the top 20 pixels to get rid of the propeller
                continue
            cul_line[int(r)] += cul_img[y + h_2, x + w_2]
            count_line[int(r)] += 1
    cul_line = cul_line / count_line
    cul_line = (cul_line - np.min(cul_line)) / (np.max(cul_line) - np.min(cul_line))
    plt.plot(cul_line, label="Photometric")
    plt.legend()
    plt.show()
    mask = np.zeros(cul_img.shape, dtype=np.float)
    for x in range(0, w):
        for y in range(0, h):
            r = int(np.sqrt((x - w_2) ** 2 + (y - h_2) ** 2))
            if r < w_2:
                mask[y, x] = cul_line[r]
    #Normalize the mask by min max to 0.0 - 1.0
    return mask
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fisheye undist')
    parser.add_argument("-i","--input", type=str, help="input bag file")
    parser.add_argument("-o","--output", type=str, default="", help="output path")
    parser.add_argument("-s","--step", type=int, default=5, help="step of stereo pair")
    parser.add_argument("-v","--verbose", action='store_true', help="show image")
    parser.add_argument("-t","--start-t", type=float, default=0, help="start time")
    args = parser.parse_args()
    #Read from bag
    bag = rosbag.Bag(args.input)
    num_imgs = bag.get_message_count("/arducam/image/compressed") + bag.get_message_count("/arducam/image")
    print("Total number of images:", num_imgs)
    bridge = CvBridge()
    pbar = tqdm.tqdm(total=num_imgs/args.step, colour="green")
    count = 0
    cul_imgs = []
    img_count = 0
    t0 = None
    for topic, msg, t in bag.read_messages():
        try:
            if t0 is None:
                t0 = t
            if (t - t0).to_sec() < args.start_t:
                continue
            if topic == "/arducam/image/compressed" or topic == "/arducam/image/raw":
                if count % args.step != 0:
                    count += 1
                    continue
                if msg._type == "sensor_msgs/Image":
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                else:
                    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                imgs = split_image(img)
                if cul_imgs == []:
                    #Create float64 image from gray image
                    gray = cv.cvtColor(imgs[0], cv.COLOR_BGR2GRAY)
                    cul_imgs = [np.zeros(gray.shape, dtype=np.float64) for img in imgs]
                for i in range(len(imgs)):
                    #Add gaussain
                    cul_imgs[i] += cv.GaussianBlur(cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY), (5, 5), 0)/255.0
                    #Show averaged image
                    if args.verbose:
                        avg = cul_imgs[i] / img_count
                        cv.imshow(f"Raw {i}", imgs[i])
                        cv.imshow(f"Average {i}", avg)
                        cv.waitKey(1)
                avg_23 = (cul_imgs[2] + cul_imgs[3]) / 2.0  / img_count
                img_count += 1
                pbar.update(1)
                if args.verbose:
                    c = cv.waitKey(1)
                    if c == ord('q'):
                        break
                count += 1
        except KeyboardInterrupt:
            break
    mask = findPhotometric((cul_imgs[2] + cul_imgs[3]) / 2.0/img_count)
    cv.imshow("Mask", mask)
    cv.imwrite(args.output, mask * 255)
    cv.waitKey(0)