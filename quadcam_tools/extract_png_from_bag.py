# Extrac png from rosbag
from utils.stereo_gen import *
from utils.fisheye_undist import *
import cv2 as cv
import numpy as np
import argparse
import rosbag
import tqdm
from cv_bridge import CvBridge
from quad_cam_split import split_image
from test_depth_estimation import calib_photometric_imgs , loadConfig, calib_photometric_imgs_individual
import os

topic_list = ["/cam_0_0/compressed", "/cam_0_1/compressed", 
        "/cam_1_0/compressed", "/cam_1_1/compressed",
        "/cam_2_0/compressed", "/cam_2_1/compressed",
        "/cam_3_0/compressed", "/cam_3_1/compressed",
        "/image_0/compressed"]

topic_counter_map = {}

def init_topic_counter():
    for topic in topic_list:
        topic_counter_map[topic] = 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract png from rosbag')
    parser.add_argument('--bag', type=str, help='path to rosbag')
    parser.add_argument('--output', type=str, help='path to output folder')
    parser.add_argument('--step', type=int, help='image extract ',default=1)
    parser.add_argument('--number', type=int, help='path to calibration file',default=-1)
    parser.add_argument('--width', type=int, help='path to calibration file') #720p = 1280 x 720
    parser.add_argument('--height', type=int, help='path to calibration file') #
    args = parser.parse_args()
    
    ouput_path = ""
    if args.output != "":
        ouput_path = args.output
        if not os.path.exists(ouput_path):
            os.makedirs(ouput_path)
            print("Created output folder: ", ouput_path)
    else:
        print("neet to specify output folder\n")
        exit(1)
    print("output path: ", ouput_path)
    
    bag_path = ""
    if args.bag != "":
        bag_path = args.bag
        if not os.path.exists(bag_path):
            print("bag file not exist\n")
            exit(1)

    num_imgs = 0
    bag = rosbag.Bag(bag_path)
    for topic in topic_list:
        num_imgs += bag.get_message_count(topic)
    print ("num_imgs: ", num_imgs)
    if(args.number ==-1 or args.number > num_imgs):
        args.number = num_imgs

    init_topic_counter()

    step = 1
    
    if args.step != "":
        if int(args.step) > 0:
            step = int(args.step)
    else:
        print("step must be positive\n")
        exit(1)

    set_width = 0
    set_height = 0
    try:
        if int(args.width) > 0:
                set_width = int(args.width)
        else:
            set_width = 320
            print("set width to default: ", set_width)
    except:
        set_width = 320
        print("set width to default: ", set_width)
    try:
        if int(args.height) > 0:
                set_height = int(args.height)
        else:
            set_height = 240
            print("set height to default: ", set_height)
    except:
        set_height = 240
        print("set height to default: ", set_height)

    
    ## Try read a frame to evalutate image size

    for topic, msg, t in bag.read_messages(topic_list[0]):
        if msg._type == "sensor_msgs/CompressedImage":
            cv_img = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            height, width, channels = cv_img.shape
            if set_width > 0 and set_height > 0:
                if set_width > width or set_height > height:
                    print("Can not resize image to define shape\n")
                    exit(1)
            sub_dir_name = topic.replace("/", "_")
            if sub_dir_name[0] == "_":
                sub_dir_name = sub_dir_name[1:]
            output_path_dir = ouput_path + "/" + sub_dir_name +"/"
            print("sub_dir_name: ", output_path_dir)
            break

    pbar = tqdm.tqdm(total= [num_imgs,args.number][num_imgs > args.number] ,colour="green")
    count = 0

    ## check image size

    for topic, msg, t in bag.read_messages():
        try:
            if topic in topic_list and count <= args.number and msg._type == "sensor_msgs/CompressedImage":
                if (topic_counter_map[topic] % step) == 0 : 
                    print("skip image: ", topic_counter_map[topic])
                    cv_img = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
                    sub_dir_name = topic.replace("/", "_")
                    if sub_dir_name[0] == "_":
                        sub_dir_name = sub_dir_name[1:]
                    output_path_dir = ouput_path + "/" + sub_dir_name +"/"
                    if os.path.exists(output_path_dir) == False:
                        os.makedirs(output_path_dir)
                    cv.imwrite(output_path_dir + str(int(topic_counter_map[topic]/step)) + ".png", cv_img)
                    topic_counter_map[topic] += 1
                    # print("save image: ", output_path_dir + str(count) + ".png")
                    pbar.update(1)
                    count += 1
                else:
                    topic_counter_map[topic] += 1
                    continue
                    
        except KeyboardInterrupt:
            break



    
