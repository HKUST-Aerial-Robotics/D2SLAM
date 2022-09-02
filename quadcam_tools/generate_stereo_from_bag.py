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
from test_depth_estimation import calib_photometric_imgs, loadConfig
import os

def genDefaultConfig():
    K = np.array([[1162.5434300524314, 0, 660.6393183718625],
        [0, 1161.839362615319,  386.1663300322095],
        [0, 0, 1]])
    D = np.array([-0.17703529535292872, 0.7517933338735744, -0.0008911425891703079, 2.1653595535258756e-05])
    xi = 2.2176903753419963
    undist = FisheyeUndist(K, D, xi, fov=args.fov)
    gen = StereoGen(undist, undist, np.eye(3), np.zeros(3))
    return [gen, gen, gen, gen]

def kablirCalibratePinhole(topic_a, topic_b, bagfile, output_calib_name, verbose=False, init_focal_length=400):
    import subprocess
    bagfile = os.path.basename(bagfile)
    bagname = bagfile.split(".")[0]
    bagpath = os.path.dirname(bagfile)
    if bagpath == "":
        bagpath = os.getcwd()
        print("bagpath", bagpath)
    cmd = f"""#!/bin/bash
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1 
export KALIBR_FOCAL_LENGTH_INIT_VALUE={init_focal_length}
source /catkin_ws/devel/setup.bash && \
rosrun kalibr kalibr_calibrate_cameras --bag /data/{bagfile} --target /data/aprilgrid.yaml --models pinhole-radtan pinhole-radtan --approx-sync 0.01 --topics {topic_a} {topic_b}"""
    if not verbose:
        cmd += " --dont-show-report"
    with open(f"{bagpath}/stereo_calib.sh", "w") as f:
        f.write(cmd)
    os.system(f"chmod +x {bagpath}/stereo_calib.sh")
    dockercmd = f"""docker run --rm -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "{bagpath}:/data" kalibr /data/stereo_calib.sh"""
    print(dockercmd)
    p_docker = subprocess.Popen(dockercmd, shell=True, stderr=subprocess.STDOUT)
    p_docker.wait()
    results = f"{bagpath}/{bagname}-results-cam.txt"
    os.rename(results, f"{bagpath}/{output_calib_name}-results.txt")
    results = f"{bagpath}/{bagname}-camchain.yaml"
    os.rename(results, f"{bagpath}/{output_calib_name}.yaml")
    print("Finished calibrate:", output_calib_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fisheye undist')
    parser.add_argument("-i","--input", type=str, help="input bag file")
    parser.add_argument("-f","--fov", type=float, default=190, help="hoizon fov of fisheye")
    parser.add_argument("-c","--config", type=str, default="", help="config file path")
    parser.add_argument("-o","--output", type=str, default="", help="output path")
    parser.add_argument("-s","--step", type=int, default=1, help="step of stereo pair")
    parser.add_argument("-v","--verbose", action='store_true', help="show image")
    parser.add_argument("-p","--photometric", type=str, help="photometric calibration image")
    parser.add_argument("-w","--width", type=int, default=600, help="width of image")
    parser.add_argument("--height", type=int, default=300, help="height of image")
    args = parser.parse_args()
    if args.config == "":
        stereo_gens = genDefaultConfig()
    else:
        stereo_gens = loadConfig(args.config, fov=args.fov, width=args.width, height=args.height)
    if args.output != "":
        output_bag = rosbag.Bag(args.output, mode="w")
    else:
        output_bag = None
    #Read photometric
    if args.photometric != "":
        print("Loading photometric calibration image")
        photometric = cv.imread(args.photometric, cv.IMREAD_GRAYSCALE)/255.0
    else:
        photometric = None
    
    #Read from bag
    bag = rosbag.Bag(args.input)
    num_imgs = bag.get_message_count("/arducam/image/compressed") + bag.get_message_count("/arducam/image")
    print("Total number of images:", num_imgs)
    bridge = CvBridge()
    pbar = tqdm.tqdm(total=num_imgs//args.step, colour="green")
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
                calibed = calib_photometric_imgs(imgs, photometric)
                for gen in stereo_gens:
                    cam_idx_a = gen.cam_idx_a
                    cam_idx_b = gen.cam_idx_b
                    idx_vcam_a = gen.idx_l
                    idx_vcam_b = gen.idx_r
                    img_l, img_r = gen.genStereo(calibed[cam_idx_a], calibed[cam_idx_b])
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
    for gen in stereo_gens:
        cam_idx_a = gen.cam_idx_a
        cam_idx_b = gen.cam_idx_b
        idx_vcam_a = gen.idx_l
        idx_vcam_b = gen.idx_r
        topic_l, topic_r = f"/cam_{cam_idx_a}_{idx_vcam_a}/compressed", f"/cam_{cam_idx_b}_{idx_vcam_b}/compressed"
        kablirCalibratePinhole(topic_l, topic_r, args.output, f"stereo_calib_{cam_idx_a}_{cam_idx_b}", verbose = args.verbose, 
                init_focal_length = stereo_gens[0].undist_l.focal_gen)
