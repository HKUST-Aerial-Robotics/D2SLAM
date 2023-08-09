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
from pathlib import Path

home = str(Path.home())

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

def initStereoFromConfig(config_file, stereo_gen, force_width=None):
    import yaml
    print("Init stereo with config", config_file)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        size = config["cam0"]["resolution"]
        K0 = pinholeIntrinsicToCameraMatrix(config["cam0"]['intrinsics'])
        K1 = pinholeIntrinsicToCameraMatrix(config["cam1"]['intrinsics'])
        if force_width is not None:
            K0 *= force_width/size[0]
            K1 *= force_width/size[0]
            size = (force_width, int(force_width*size[1]/size[0]))
            print("Force width to", force_width, "\nK0\n", K0, "\nK1\n", K1)
        D0 = np.array(config["cam0"]['distortion_coeffs'], dtype=np.float)
        D1 = np.array(config["cam1"]['distortion_coeffs'], dtype=np.float)
        T = np.array(config["cam1"]["T_cn_cnm1"])
        R = T[0:3,0:3]
        t = T[0:3,3]
        stereo_gen.initRectify(K0, D0, K1, D1, (size[0], size[1]), R, t)

def loadHitnet():
    print("Loading hitnet...")
    HITNET_PATH = '/home/xuhao/source/ONNX-HITNET-Stereo-Depth-estimation/'
    sys.path.insert(0, HITNET_PATH)
    from hitnet import HitNet, ModelType
    model_path = HITNET_PATH + "/models/eth3d/saved_model_240x320/model_float32.onnx"
    return HitNet(model_path, ModelType.eth3d), False

def loadCRENet():
    print("Loading hitnet...")
    CRENETPath = '/home/xuhao/source/ONNX-CREStereo-Depth-Estimation'
    CRENETPath = '/home/dji/source/ONNX-CREStereo-Depth-Estimation'
    sys.path.insert(0, CRENETPath)
    from crestereo import CREStereo
    iters = 5            # Lower iterations are faster, but will lower detail. 
    shape = (240, 320)   # Input resolution. 
    version = "combined" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
    model_path = CRENETPath+f'/models/crestereo_{version}_iter{iters}_{shape[0]}x{shape[1]}.onnx'
    depth_estimator = CREStereo(model_path)
    return depth_estimator, True

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
            try:
                T = np.array(config[v]['T_cam_imu'])
            except: 
                T = np.eye(4)
            undist = FisheyeUndist(K, D, xi, fov=fov, width=width, height=height, extrinsic=T)
            undists.append(undist)
    hitnet_model, is_rgb = loadCRENet() if hitnet else (None, False)
    # gens = [StereoGen(undists[1], undists[0], cam_idx_a=1,  cam_idx_b=0, hitnet_model=hitnet_model, is_rgb=is_rgb),
    #         StereoGen(undists[2], undists[1], cam_idx_a=2,  cam_idx_b=1, hitnet_model=hitnet_model, is_rgb=is_rgb),
    #         StereoGen(undists[3], undists[2], cam_idx_a=3,  cam_idx_b=2, hitnet_model=hitnet_model, is_rgb=is_rgb),
    #         StereoGen(undists[0], undists[3], cam_idx_a=0,  cam_idx_b=3, hitnet_model=hitnet_model, is_rgb=is_rgb)]
    
    gens = [StereoGen(undists[0], undists[1], cam_idx_a=0,  cam_idx_b=1, hitnet_model=hitnet_model, is_rgb=is_rgb),
        StereoGen(undists[1], undists[2], cam_idx_a=1,  cam_idx_b=2, hitnet_model=hitnet_model, is_rgb=is_rgb),
        StereoGen(undists[2], undists[3], cam_idx_a=2,  cam_idx_b=3, hitnet_model=hitnet_model, is_rgb=is_rgb),
        StereoGen(undists[3], undists[0], cam_idx_a=3,  cam_idx_b=0, hitnet_model=hitnet_model, is_rgb=is_rgb)]
    for i in range(len(config_stereos)):
        initStereoFromConfig(config_stereos[i], gens[i], force_width=width)
    return gens, is_rgb

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

count = 0

def test_depth_gen(gen: StereoGen, imgs_calib, imgs_raw, detailed=False, save_rgb=True):
    cam_idx_a = gen.cam_idx_a
    cam_idx_b = gen.cam_idx_b
    idx_vcam_a = gen.idx_l
    idx_vcam_b = gen.idx_r
    global count
    if detailed:
        disparity = gen.genDisparity(imgs_calib[cam_idx_a], imgs_calib[cam_idx_b])
        texture = gen.rectifyL(imgs_raw[cam_idx_a])
        #Visualize disparity
        disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # disparity = (disparity * 255.0/32.0).astype(np.uint8)
        disparity = cv.applyColorMap(disparity, cv.COLORMAP_JET)
        cv.rectangle(disparity, gen.roi_l, (0, 0, 255), 2)
        if count == 0:
            Path(home+"/output/stereo_calib/").mkdir(parents=True, exist_ok=True)

        img_l, img_r = gen.genRectStereo(imgs_raw[cam_idx_a], imgs_raw[cam_idx_b])
        cv.imwrite(home+f"/output/stereo_calib/left_{count}.png", img_l)
        cv.imwrite(home+f"/output/stereo_calib/right_{count}.png", img_r)

        img_l, img_r = gen.genRectStereo(imgs_calib[cam_idx_a], imgs_calib[cam_idx_b])
        cv.imwrite(home+f"/output/stereo_calib/gray_left_{count}.png", img_l)
        cv.imwrite(home+f"/output/stereo_calib/gray_right_{count}.png", img_r)

        
        count += 1
        if len(img_l.shape) == 2:
            img_l = cv.cvtColor(img_l, cv.COLOR_GRAY2BGR)
        if len(img_r.shape) == 2:
            img_r = cv.cvtColor(img_r, cv.COLOR_GRAY2BGR)
        img_show = cv.hconcat([img_l, img_r, disparity])
        img_show = drawHorizontalLines(img_show)
        cv.imshow(f"stereoRect {cam_idx_a}_{idx_vcam_a} <-> {cam_idx_b}_{idx_vcam_b} hor", img_show)
        # cv.imshow(f"disp {cam_idx_a}_{idx_vcam_a} <-> {cam_idx_b}_{idx_vcam_b}", disparity)
        cv.waitKey(10)
    pcl, texture = gen.genPointCloud(imgs_calib[cam_idx_a], imgs_calib[cam_idx_b], img_raw=imgs_raw[cam_idx_a])
    return pcl, texture

def calib_photometric(img, photometric, is_rgb=True):
    if not is_rgb:
        ret = img.copy()
        if len(img.shape) == 3:
            ret = cv.cvtColor(ret, cv.COLOR_BGR2GRAY)
        ret = ret.astype(float)/photometric
    else:
        #Divide by photometric per channel
        ret = img.copy().astype(float)
        for i in range(img.shape[2]):
            ret[:,:,i] = ret[:,:,i]/photometric*0.7
    ret = np.clip(ret, 0, 255).astype(np.uint8)
    return ret

def calib_photometric_imgs(imgs, photometric, is_rgb=True):
    photometric_calibed = []
    if photometric is not None:
        #Convert to grayscale
        for img in imgs:
            calibed = calib_photometric(img, photometric, is_rgb=is_rgb)
            photometric_calibed.append(calibed)
    else:
        photometric_calibed = imgs
    return photometric_calibed
                    
def calib_photometric_imgs_individual(imgs, photometrics, is_rgb=True):
    photometric_calibed = []
    if photometrics is not None:
        #Convert to grayscale
        for i in range(len(imgs)):
            calibed = calib_photometric(imgs[i], photometrics[i], is_rgb=is_rgb)
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
    parser.add_argument("-p","--photometric", type=str, default="", help="photometric calibration image")
    parser.add_argument("-w","--width", type=int, default=320, help="width of stereo pair")
    parser.add_argument("--height", type=int, default=240, help="width of stereo pair")

    args = parser.parse_args()
    stereo_paths = ["/home/xuhao/Dropbox/data/d2slam/quadcam2/stereo_calib_1_0.yaml",
                "/home/xuhao/Dropbox/data/d2slam/quadcam2/stereo_calib_2_1.yaml",
                "/home/xuhao/Dropbox/data/d2slam/quadcam2/stereo_calib_3_2.yaml",
                "/home/xuhao/Dropbox/data/d2slam/quadcam2/stereo_calib_0_3.yaml"]
    if args.config == "":
        stereo_gens = genDefaultConfig()
    else:
        stereo_gens, is_rgb = loadConfig(args.config, stereo_paths, fov=args.fov, hitnet=True, width=args.width, height=args.height)
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
                if photometric is not None:
                    photometric_calibed = calib_photometric_imgs(imgs, photometric, is_rgb)
                    if is_rgb:
                        imgs = photometric_calibed
                    else:
                        imgs = calib_photometric_imgs(imgs, photometric, True)
                else:
                    photometric_calibed = imgs
                cv.imshow("raw", imgs[2])
                #Apply inverse of photometric calibration
                cv.imshow("Photometic calibed", photometric_calibed[2])
                cv.waitKey(1)
                for gen in stereo_gens:
                    test_depth_gen(gen, photometric_calibed, imgs, detailed=True)
                pbar.update(1)
                c = cv.waitKey(0)
                if c == ord('q'):
                    break
                count += 1
        except KeyboardInterrupt:
            break
