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
    gen = StereoGen(undist, undist)
    return [gen, gen, gen, gen]

def pinholeIntrinsicToCameraMatrix(int):
    K0 = np.array([[int[0], 0, int[2]], 
                [0, int[1], int[3]],
                [0, 0, 1]])
    return K0

def initStereoFromConfig(config_file, stereo_gen):
    import yaml
    print("Init stereo with config", config_file)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        K0 = pinholeIntrinsicToCameraMatrix(config["cam0"]['intrinsics'])
        D0 = np.array(config["cam0"]['distortion_coeffs'], dtype=np.float)
        K1 = pinholeIntrinsicToCameraMatrix(config["cam1"]['intrinsics'])
        D1 = np.array(config["cam1"]['distortion_coeffs'], dtype=np.float)
        size = config["cam0"]["resolution"]
        T = np.array(config["cam1"]["T_cn_cnm1"])
        R = T[0:3,0:3]
        t = T[0:3,3]
        stereo_gen.initRectify(K0, D0, K1, D1, (size[0], size[1]), R, t)

def loadConfig(config_file, config_stereos=[], fov=190, width=600, height=300, hitnet=False):
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
            T = np.array(config[v]['T_cam_imu'])
            undist = FisheyeUndist(K, D, xi, fov=fov, width=width, height=height, extrinsic=T)
            undists.append(undist)
    gens = [StereoGen(undists[1], undists[0], cam_idx_a=1,  cam_idx_b=0, enable_hitnet=hitnet),
            StereoGen(undists[2], undists[1], cam_idx_a=2,  cam_idx_b=1, enable_hitnet=hitnet),
            StereoGen(undists[3], undists[2], cam_idx_a=3,  cam_idx_b=2, enable_hitnet=hitnet),
            StereoGen(undists[0], undists[3], cam_idx_a=0,  cam_idx_b=3, enable_hitnet=hitnet)]
    for i in range(len(config_stereos)):
        initStereoFromConfig(config_stereos[i], gens[i])
    return gens

def drawVerticalLines(img, num_lines=10):
    #Cvt color if gray
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    h, w, _ = img.shape
    for i in range(num_lines):
        cv.line(img, (int(w/num_lines*i), 0), (int(w/num_lines*i), h), (0, 255, 0), 1)
    return img

def drawHorizontalLines(img, num_lines=10):
    #Cvt color if gray
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    h, w, _ = img.shape
    for i in range(num_lines):
        cv.line(img, (0, int(h/num_lines*i)), (w, int(h/num_lines*i)), (0, 255, 0), 1)
    return img

def drawPointCloud3d(pcl):
    #Using opencv viz to draw point cloud
    myWindow = cv.viz_Viz3d("Coordinate Frame")
    myWindow.showWidget("Coordinate Widget", cv.viz_WCoordinateSystem())
    cloud = cv.viz_WCloud(pcl)
    myWindow.showWidget("Cloud Widget", cloud)
    myWindow.spin()

def test_depth_gen(gen: StereoGen, imgs_gray, imgs_raw, detailed=False):
    cam_idx_a = gen.cam_idx_a
    cam_idx_b = gen.cam_idx_b
    idx_vcam_a = gen.idx_l
    idx_vcam_b = gen.idx_r
    if detailed:
        disparity = gen.genDisparity(imgs_gray[cam_idx_a], imgs_gray[cam_idx_b])
        texture = gen.rectifyL(imgs_raw[cam_idx_a])
        #Visualize disparity
        disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        disparity = cv.applyColorMap(disparity, cv.COLORMAP_JET)

        img_l, img_r = gen.genRectStereo(imgs_gray[cam_idx_a], imgs_gray[cam_idx_b])
        img_l = cv.cvtColor(img_l, cv.COLOR_GRAY2BGR)
        img_r = cv.cvtColor(img_r, cv.COLOR_GRAY2BGR)
        cv.imwrite("/home/xuhao/output/rect_l.png", img_l)
        cv.imwrite("/home/xuhao/output/rect_r.png", img_r)
        img_show = cv.hconcat([img_l, img_r, disparity])
        img_show = drawHorizontalLines(img_show)
        cv.imshow(f"stereoRect {cam_idx_a}_{idx_vcam_a} <-> {cam_idx_b}_{idx_vcam_b} hor", img_show)
        cv.imshow(f"disp {cam_idx_a}_{idx_vcam_a} <-> {cam_idx_b}_{idx_vcam_b}", disparity)
        cv.waitKey(10)
    pcl, texture = gen.genPointCloud(imgs_gray[cam_idx_a], imgs_gray[cam_idx_b], img_raw=imgs_raw[cam_idx_a])
    return pcl, texture

def calib_photometric(img, photometric):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.astype(float)/photometric
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def calib_photometric_imgs(imgs, photometric):
    photometric_calibed = []
    if photometric is not None:
        #Convert to grayscale
        for img in imgs:
            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            calibed = calib_photometric(img, photometric)
            photometric_calibed.append(calibed)
    else:
        photometric_calibed = imgs
    return photometric_calibed
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fisheye undist')
    parser.add_argument("-i","--input", type=str, help="input bag file")
    parser.add_argument("-f","--fov", type=float, default=190, help="hoizon fov of fisheye")
    parser.add_argument("-c","--config", type=str, default="", help="config file path")
    parser.add_argument("-s","--step", type=int, default=5, help="step of stereo pair")
    parser.add_argument("-v","--verbose", action='store_true', help="show image")
    parser.add_argument("-p","--photometric", type=str, help="photometric calibration image")
    args = parser.parse_args()
    stereo_paths = ["/home/xuhao/Dropbox/data/d2slam/quadcam2/stereo_calib_1_0.yaml",
                "/home/xuhao/Dropbox/data/d2slam/quadcam2/stereo_calib_2_1.yaml",
                "/home/xuhao/Dropbox/data/d2slam/quadcam2/stereo_calib_3_2.yaml",
                "/home/xuhao/Dropbox/data/d2slam/quadcam2/stereo_calib_0_3.yaml"]
    if args.config == "":
        stereo_gens = genDefaultConfig()
    else:
        stereo_gens = loadConfig(args.config, stereo_paths, fov=args.fov)
    #Read photometric
    if args.photometric != "":
        photometric = cv.imread(args.photometric, cv.IMREAD_GRAYSCALE)/255.0
    else:
        photometric = None

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
                photometric_calibed = []
                for img in imgs:
                    cv.imshow("raw", img)
                    #Apply inverse of photometric calibration
                    photometric_calibed = calib_photometric_imgs(imgs, photometric)
                    cv.imshow("Photometic calibed", img)
                    cv.waitKey(1)
                for gen in stereo_gens[2:3]:
                    test_depth_gen(gen, photometric_calibed, imgs, detailed=True)
                pbar.update(1)
                c = cv.waitKey(0)
                if c == ord('q'):
                    break
                count += 1
        except KeyboardInterrupt:
            break