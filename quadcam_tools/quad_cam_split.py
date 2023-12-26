#!/usr/bin/env python3
import rosbag
from os.path import exists
from cv_bridge import CvBridge
import cv2 as cv
import tqdm
import numpy as np

def generate_bagname(bag, comp=False):
    from pathlib import Path
    p = Path(bag)
    bagname = p.stem + "-split.bag"
    output_bag = p.parents[0].joinpath(bagname)
    return output_bag

def split_image(img, num_subimages = 4):
    #Split image vertically
    h, w = img.shape[:2]
    sub_w = w // num_subimages
    sub_imgs = []
    for i in range(num_subimages):
        sub_imgs.append(img[:, i*sub_w:(i+1)*sub_w])
    return sub_imgs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split quadcam images')
    parser.add_argument("-i","--input", type=str, help="input bag file")
    parser.add_argument('-v', '--show', action='store_true', help='compress the image topics')
    parser.add_argument('-s', '--step', type=int, nargs="?", help="step for images, default 1", default=1)
    parser.add_argument('-t', '--start', type=float, nargs="?", help="start time of the first image, default 0", default=0)
    args = parser.parse_args()
    output_bag = generate_bagname(args.input)
    if not exists(args.input):
        print(f"Input bag file {args.input} does not exist")
        exit(1)
    
    bag = rosbag.Bag(args.input)
    num_imgs = bag.get_message_count("/arducam/image/compressed") + bag.get_message_count("/arducam/image") + \
        bag.get_message_count("/oak_ffc_4p/assemble_image/compressed") + bag.get_message_count("/oak_ffc_4p/assemble_image")
    print("Total number of images:", num_imgs)
    bridge = CvBridge()

    pbar = tqdm.tqdm(total=num_imgs/args.step, colour="green")
    with rosbag.Bag(output_bag, 'w') as outbag:
        from nav_msgs.msg import Path
        path = Path()
        path_arr = []
        c = 0
        t0 = None
        for topic, msg, t in bag.read_messages():
            if t0 is None:
                t0 = t
            if (t - t0).to_sec() < args.start:
                continue
            if topic == "/arducam/image/compressed" or topic == "/arducam/image/raw" or \
                topic == "/oak_ffc_4p/assemble_image/compressed" or topic == "/oak_ffc_4p/assemble_image":
                c += 1
                if c % args.step != 0:
                    continue
                if msg._type == "sensor_msgs/Image":
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                else:
                    img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
                imgs = split_image(img)
                #Compress and write imgs to output bag
                for i, _img in enumerate(imgs):
                    # if i == 3:
                    comp_img = bridge.cv2_to_compressed_imgmsg(_img)
                    comp_img.header = msg.header
                    outbag.write(f"/d2slam/image_{i}/compressed", comp_img, t)
                    # cv.imwrite(f"/home/xuhao/output/quadvins-output/imgs/fisheye_{c:06d}_{i}.jpg", _img)
                if args.show:
                    for i in range(len(imgs)):
                        cv.imshow(f"{topic}-{i}", imgs[i])
                    cv.imshow(topic, img)
                    cv.waitKey(1)
                # Update progress bar
                pbar.update(1)
            else:
                outbag.write(topic, msg, t)
                


