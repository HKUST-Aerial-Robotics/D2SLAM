#!/usr/bin/env python3

#how to use  python3 ./generate_stereo_from_bag.py --input /media/khalil/ssd_data/data_set/omni-pinhole/camera_stereo_calib/bag/omni_calibration_2023-07-31-23-34-55.bag --output /media/khalil/ssd_data/data_set/omni-pinhole-0906  --fov 190 --config ../config/new_quadcam_NO_0/quad_cam_calib-camchain-imucam-new-3.yaml   --photometric ../config/new_quadcam_NO_0/camera_vig_mask --width 160 --height 120
#pinhole   you need to prepare folder

import utils.config_loader as config_loader
import utils.photometric_calibration as photometric_calibration
import utils.split_image as split_image
import multiprocessing
import cv2 as cv
import numpy as np
import argparse
import rosbag
import tqdm
from cv_bridge import CvBridge
import os
import shutil

# set photometric mask before run this tool
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

def IndividualPhotometricCalibration(imgs,photometric_dir_path, is_rgb=True):
    photometric_calibed = []
    photometric_imgs = []
    if os.path.exists(photometric_dir_path):
        # print("reading photometric calibration images with name cam_x_vig_mask.png")
        for i in range(len(imgs)):
            photo_metric_cali_img = cv.imread(photometric_dir_path + "/cam_" + str(i) + "_vig_mask.png", cv.IMREAD_GRAYSCALE)/255.0
            photometric_imgs.append(photo_metric_cali_img)
        for i in range(len(imgs)):
            calibed = calib_photometric(imgs[i], photometric_imgs[i], is_rgb=is_rgb)
            photometric_calibed.append(calibed)
    else:
        photometric_calibed = imgs
    return photometric_calibed

def kablirCalibratePinhole(topic_a, topic_b, bagfile, output_calib_name, verbose=False, init_focal_length=400):
    import subprocess
    #debug
    print("topic_a", topic_a)
    print("topic_b", topic_b)
    print("bagfile", bagfile)
    print("output_calib_name", output_calib_name)
    print("verbose", verbose)
    print("init_focal_length", init_focal_length)
    print("debug end")

    bagname = os.path.basename(bagfile)
    bagpath = os.path.dirname(bagfile)
    print("[DEBUG]bagpath xxxxxx", bagpath)

    if bagpath == "":
        bagpath = os.getcwd()
        print("bagpath", bagpath)
    cmd = f"""#!/bin/bash
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
export KALIBR_FOCAL_LENGTH_INIT_VALUE={init_focal_length}
source /catkin_ws/devel/setup.bash &&
rosrun kalibr kalibr_calibrate_cameras --bag /data/{bagname} --target /data/aprilgrid.yaml --models pinhole-radtan pinhole-radtan --approx-sync 0.01 --topics {topic_a} {topic_b}"""
    if not verbose:
        cmd += " --dont-show-report"
    cmd += f"""<<EOF
{init_focal_length}
{init_focal_length}
EOF"""
    with open(f"{bagpath}/stereo_calib.sh", "w") as f:
        f.write(cmd)
    os.system(f"chmod +x {bagpath}/stereo_calib.sh")
    # dockercmd = f"""docker run  -it --rm -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" \
    # -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    # -v "{bagpath}:/data" kalibr:latest /bin/bash /data/stereo_calib.sh"""
    dockercmd = f"""docker run --rm -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" --entrypoint="/data/stereo_calib.sh" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "{bagpath}:/data" kalibr:latest  """
    print(dockercmd)
    p_docker = subprocess.Popen(dockercmd, shell=True, stderr=subprocess.STDOUT)
    p_docker.wait()
    calibration_resualt_title = bagname.split(".")[0]
    print("calibration_resualt_title", calibration_resualt_title)
    results = f"{bagpath}/{calibration_resualt_title}-results-cam.txt"
    os.rename(results, f"{bagpath}/{output_calib_name}-results.txt")
    results = f"{bagpath}/{calibration_resualt_title}-camchain.yaml"
    os.rename(results, f"{bagpath}/{output_calib_name}.yaml")
    results = f"{bagpath}/{calibration_resualt_title}-report-cam.pdf"
    os.rename(results, f"{bagpath}/{output_calib_name}-report-cam.pdf")
    print("Finished calibrate:", output_calib_name)

def calibration_task(calibration_gen, stereo_calib_gens, input_bag_path, height, width, verbose):
    retries = 0
    max_retries = 3
    output_calib_name = f"stereo_calib_{calibration_gen.cam_idx_a}_{calibration_gen.cam_idx_b}_{height}_{width}"
    calib_bag_path = os.path.join(os.path.dirname(input_bag_path), output_calib_name + '.bag')
    report_pdf_path = os.path.join(os.path.dirname(calib_bag_path), f"{output_calib_name}-report-cam.pdf")
    while retries < max_retries:
        try:
            if not os.path.exists(report_pdf_path):
                topic_l = f"/cam_{calibration_gen.cam_idx_a}_{calibration_gen.idx_l}/compressed"
                topic_r = f"/cam_{calibration_gen.cam_idx_b}_{calibration_gen.idx_r}/compressed"
                shutil.copy2(input_bag_path, calib_bag_path)
                kablirCalibratePinhole(topic_l, topic_r, calib_bag_path, output_calib_name, verbose=verbose,
                                    init_focal_length=stereo_calib_gens[0].undist_l.focal_gen)
                os.remove(calib_bag_path)
            else:
                break
        except KeyboardInterrupt:
            os.remove(calib_bag_path)
            print("Ctrl+C detected. Exiting.")
            break
        except Exception as e:
            os.remove(calib_bag_path)
            print(e)
            retries += 1
            if retries < max_retries:
                print(f"Retrying... ({retries}/{max_retries})")
            else:
                print("Max retries reached. Exiting.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fisheye undist')
    parser.add_argument("-i","--input", type=str, help="input bag file")
    parser.add_argument("-f","--fov", type=float, default=190, help="hoizon fov of fisheye")
    parser.add_argument("-c","--config", type=str, default="", help="config file path")
    parser.add_argument("-o","--output", type=str, default="", help="output path")
    parser.add_argument("-s","--step", type=int, default=1, help="step of stereo pair")
    parser.add_argument("-v","--verbose", action='store_true', help="show image")
    parser.add_argument("-p","--photometric", type=str, help="photometric calibration images path")
    parser.add_argument("-w","--width", type=int, default=600, help="width of image")
    parser.add_argument("--height", type=int, default=300, help="height of image")
    args = parser.parse_args()
    if args.config == "":
        print("[INPUT ERROR] shoudl provide all fisheye cameras intrinsic and extrinsic parameters")
        exit(1)
    else:
        stereo_gens, _ = config_loader.LoadFisheyeParameter(args.config, fov=args.fov, width=args.width, height=args.height)
    if args.output != "":
        output_path = args.output
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print("copy aprilgrid.yaml to output folder")
            print(f"{output_path} created")
        aprilgrid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils/aprilgrid.yaml")
        shutil.copyfile(aprilgrid_path, os.path.join(output_path, "aprilgrid.yaml"))
        print("copy aprilgrid.yaml to output folder")
        output_bag_name = os.path.join(output_path, "stereo_calibration_step_%s_width_%s_height_%s.bag" % (args.step, args.width, args.height))
    else:
        output_bag = None

    print("[Debug] ouputbag", args.output)
    #Read photometric
    photometrics = []
    if args.photometric:
        print("Loading photometric calibration images from path:", args.photometric)
        for i in range(4):
            vig_png_name = args.photometric +  f"/cam_{i}_vig_mask.png"
            if not os.path.exists(vig_png_name):
                print(f"vig_png_name {vig_png_name} does not exist")
                exit(1)
            vig_cali_pic = cv.imread(vig_png_name, cv.IMREAD_GRAYSCALE)/255.0
            photometrics.append(vig_cali_pic)
        # photometrics = cv.imread(args.photometric, cv.IMREAD_GRAYSCALE)/255.0
    else:
        photometrics = None

    # If bag already exists, no need to generate again
    if not os.path.exists(output_bag_name):
        #Read from bag
        output_bag = rosbag.Bag(output_bag_name, mode="w")
        bag = rosbag.Bag(args.input)
        num_imgs = bag.get_message_count("/arducam/image/compressed") + bag.get_message_count("/arducam/image") + \
            bag.get_message_count("/oak_ffc_4p/assemble_image/compressed") + bag.get_message_count("/oak_ffc_4p/assemble_image")
        print("Total number of images:", num_imgs)
        bridge = CvBridge()
        pbar = tqdm.tqdm(total=num_imgs//args.step, colour="green")
        count = 0
        for topic, msg, t in bag.read_messages():
            try:
                if topic == "/arducam/image/compressed" or topic == "/arducam/image/raw" \
                        or topic == "/oak_ffc_4p/assemble_image/compressed" or topic == "/oak_ffc_4p/assemble_image":
                    if count % args.step != 0:
                        count += 1
                        continue
                    if msg._type == "sensor_msgs/Image":
                        img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    else:
                        img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    imgs = split_image.splitImage(img)
                    calibed = photometric_calibration.calibPhotometricImgsIndividual(imgs, photometrics, is_rgb=False)
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
    else:
        print('Using the previours bag: %s' % str(output_bag_name))

    ### I prefer do it in docker
    # Create a multiprocessing pool with 4 processes
    try:
        pool = multiprocessing.Pool(processes=4)
        for gen in stereo_gens:
            # Apply the calibration_task function to each generator in parallel
            pool.apply_async(calibration_task, args=(gen, stereo_gens, output_bag_name, args.height, args.width, args.verbose,), error_callback=print)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        # Catch Ctrl+C
        # TODO: Ctrl+C still not working
        print("Ctrl+C detected. Stopping all tasks...")
        pool.terminate()
        pool.join()