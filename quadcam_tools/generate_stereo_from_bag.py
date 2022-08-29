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

def genDefaultConfig():
    K = np.array([[1162.5434300524314, 0, 660.6393183718625],
        [0, 1161.839362615319,  386.1663300322095],
        [0, 0, 1]])
    D = np.array([-0.17703529535292872, 0.7517933338735744, -0.0008911425891703079, 2.1653595535258756e-05])
    xi = 2.2176903753419963
    undist = FisheyeUndist(K, D, xi, fov=args.fov)
    gen = StereoGen(undist, undist, np.eye(3), np.zeros(3))
    return [gen, gen, gen, gen]

def loadConfig(config_file):
    print("Loading config from", config_file)
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
    gens = [StereoGen(undists[1], undists[0], cam_idx_a=1,  cam_idx_b=0),
            StereoGen(undists[2], undists[1], cam_idx_a=2,  cam_idx_b=1),
            StereoGen(undists[3], undists[2], cam_idx_a=3,  cam_idx_b=2),
            StereoGen(undists[0], undists[3], cam_idx_a=0,  cam_idx_b=3)]
    return gens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fisheye undist')
    parser.add_argument("-i","--input", type=str, help="input bag file")
    parser.add_argument("-f","--fov", type=float, default=190, help="hoizon fov of fisheye")
    parser.add_argument("-c","--config", type=str, default="", help="config file path")
    parser.add_argument("-o","--output", type=str, default="", help="output path")
    parser.add_argument("-s","--step", type=int, default=5, help="step of stereo pair")
    parser.add_argument("-v","--verbose", action='store_true', help="show image")
    parser.add_argument("--calib-phom", action='store_true', help="calib-phom")
    args = parser.parse_args()
    if args.config == "":
        stereo_gens = genDefaultConfig()
    else:
        stereo_gens = loadConfig(args.config)
    if args.output != "":
        output_bag = rosbag.Bag(args.output, mode="w")
    else:
        output_bag = None
    #Read from bag
    bag = rosbag.Bag(args.input)
    num_imgs = bag.get_message_count("/arducam/image/compressed") + bag.get_message_count("/arducam/image")
    print("Total number of images:", num_imgs)
    bridge = CvBridge()
    pbar = tqdm.tqdm(total=num_imgs/args.step, colour="green")
    count = 0
    for topic, msg, t in bag.read_messages():
        try:
            if topic == "/arducam/image/compressed" or topic == "/arducam/image/raw":
                if count % args.step != 0:
                    count += 1
                    continue
                if msg._type == "sensor_msgs/Image":
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                else:
                    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                imgs = split_image(img)
                # if args.calib_phom:
                #     img_l, img_r = stereo_gens[3].calibPhotometric(imgs[3], imgs[2])
                #     img_show = cv.hconcat([img_l, img_r])
                #     cv.imshow("calibPhotometric", img_show)
                for gen in stereo_gens:
                    cam_idx_a = gen.cam_idx_a
                    cam_idx_b = gen.cam_idx_b
                    idx_vcam_a = gen.idx_l
                    idx_vcam_b = gen.idx_r
                    img_l, img_r = gen.genStereo(imgs[cam_idx_a], imgs[cam_idx_b])
                    img_show = cv.hconcat([img_l, img_r])
                    if args.verbose:
                        cv.imshow(f"stereo {cam_idx_a}_{idx_vcam_a} <-> {cam_idx_b}_{idx_vcam_b}", img_show)
                    topic_l, topic_r = f"/cam_{cam_idx_a}_{idx_vcam_a}/compressed", f"/cam_{cam_idx_b}_{idx_vcam_b}/compressed"
                    comp_img_l, comp_img_r = bridge.cv2_to_compressed_imgmsg(img_l), bridge.cv2_to_compressed_imgmsg(img_r)
                    comp_img_l.header = comp_img_r.header = msg.header
                    output_bag.write(topic_l, comp_img_l, t)
                    output_bag.write(topic_r, comp_img_r, t)

                pbar.update(1)
                if args.verbose:
                    c = cv.waitKey(1)
                    if c == ord('q'):
                        break
                count += 1
        except KeyboardInterrupt:
            break
    output_bag.close()