#!/usr/bin/env python3
from transformations import *
from math import *
from cv_bridge import CvBridge
import cv2 as cv
import argparse
from sensor_msgs.msg import CompressedImage
import pathlib
import rosbag

bridge = CvBridge()
first_print = False

def compress_image_msg(msg, resize=0.4):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cv_image = cv.resize(cv_image, (0, 0), fx=resize, fy=resize)
    comp_img = CompressedImage()
    comp_img.header = msg.header
    comp_img.format = "mono8; jpeg compressed"
    succ, _data = cv.imencode(".jpg", cv_image, encode_param)
    comp_img.data = _data.flatten().tolist()
    global first_print
    if not first_print:
        print(f"Compress image from {msg.height}x{msg.width} to {cv_image.shape}")
        first_print = True
    return comp_img, cv_image

def generate_bagname(bag, output_path, comp=False):
    from pathlib import Path
    p = Path(bag)
    if comp:
        bagname = p.stem + "-resize-comp.bag"
    else:
        bagname = p.stem + "-resize.bag"
    output_bag = output_path.joinpath(bagname)
    return output_bag

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bags', metavar='bags', type=str, nargs='+',
                    help='bags to be synchronized')
    parser.add_argument('-c', '--comp', action='store_true', help='compress the image topics', default=True)
    parser.add_argument('-q', '--quality', type=int, default=90, help='quality of the compressed image')
    parser.add_argument('-r', '--resize', type=float, default=0.4, help='resize scale of the image')
    parser.add_argument('-s', '--show', action='store_true', help="show while sync")
    parser.add_argument('-o', '--output', default="", type=str, help='output path')
    args = parser.parse_args()
    bags = args.bags

    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), args.quality]

    output_path = pathlib.Path(bags[0]).parents[0] if args.output == "" else pathlib.Path(args.output)
    print(f"{len(bags)} bags to process. Will write to", output_path)
    for bag in bags:
        output_bag = generate_bagname(bag, output_path, args.comp)
        print("Write bag to", output_bag)
        with rosbag.Bag(output_bag, 'w') as outbag:
            for topic, msg, t in rosbag.Bag(bag).read_messages():
                if msg._type == "sensor_msgs/Image" and args.comp:
                    comp_img, cv_image = compress_image_msg(msg, args.resize)
                    outbag.write(topic+"_resize/compressed", comp_img, t)
                    if args.show:
                        # Show timestamp, size on the imageimage
                        cv.putText(cv_image, f"{t.to_sec():.3f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv.putText(cv_image, f"{cv_image.shape[1]}x{cv_image.shape[0]}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv.imshow(topic, cv_image)
                        cv.waitKey(1)
                else:
                    outbag.write(topic, msg, t)
                    