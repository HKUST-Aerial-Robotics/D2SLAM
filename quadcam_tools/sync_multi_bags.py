#!/usr/bin/env python3
import rosbag
from transformations import *
import numpy as np
from math import *
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2 as cv
import argparse
from sensor_msgs.msg import CompressedImage
import rospy

def quat2eulers(w, x, y, z):
    r = atan2(2 * (w * x + y * z),
                    1 - 2 * (x * x + y * y))
    p = asin(2 * (w * y - z * x))
    y = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return y, p, r

def yaw_rotate_vec(yaw, vec):
    Re = rotation_matrix(yaw, [0, 0, 1])[0:3, 0:3]
    return np.transpose(np.dot(Re, np.transpose(vec)))

def generate_bagname(bag, output_path, comp=False):
    from pathlib import Path
    p = Path(bag)
    if comp:
        bagname = p.stem + "-sync-comp.bag"
    else:
        bagname = p.stem + "-sync.bag"
    output_bag = output_path.joinpath(bagname)
    # output_bag = "/home/xuhao/Dropbox/data/d2slam/tum_datasets/" + bagname
    return output_bag

def generate_groundtruthname(bag):
    from pathlib import Path
    p = Path(bag)
    bagname = p.stem + "-groundtruth.txt"
    output_bag = p.parents[0].joinpath(bagname)
    return output_bag

def get_time0(bag, is_realsense=False):
    count_camera_available = set()
    for topic, msg, t in rosbag.Bag(bag).read_messages():
        # We use image time as the start time
        if msg._type == "sensor_msgs/Image" or msg._type == "sensor_msgs/CompressedImage":
            if not is_realsense:
                return t
            if topic == "/camera/infra1/image_rect_raw/compressed" or topic == "/camera/infra1/image_rect_raw":
                count_camera_available.add(1)
            if topic == "/camera/infra2/image_rect_raw/compressed" or topic == "/camera/infra2/image_rect_raw":
                count_camera_available.add(2)
            if len(count_camera_available) == 2:
                return t
mav = None
def get_traj_command_time(bag):
    from pymavlink4swarm import MAVLink
    import array
    class fifo(object):
        def __init__(self):
            self.buf = []
        def write(self, data):
            self.buf += data
            return len(data)
        def read(self):
            return self.buf.pop(0)
    global mav
    if mav is None:
        f = fifo()
        mav = MAVLink(f)
    print("Processing...")
    for topic, msg, t in rosbag.Bag(bag).read_messages():
        if topic == "/uwb_node/incoming_broadcast_data":
            m = mav.decode(array.array('B', (msg.data)))
            if m.command_type == 16:
                print(f"Got command {m.command_type} at {t}")
                return t
    return None

def get_pose0(bag):
    for topic, msg, t in rosbag.Bag(bag).read_messages():
        if topic == "/vrpn_client/raw_transform":
            quat0 = msg.transform.rotation
            pos0 = msg.transform.translation
            y0, p0, r0 = quat2eulers(quat0.w, quat0.x, quat0.y, quat0.z)
            pos0 = np.array([pos0.x, pos0.y, pos0.z])
            q_calib = quaternion_from_euler(0, 0, -y0)
            # ypr = 
            print(f"Will use {t0} as start yaw0 {y0} pos0 {pos0} qcalib {q_calib}")
            return pos0, q_calib, y0
        elif topic == "/SwarmNode1/pose" or topic=="/leica/pose/relative":
            quat0 = msg.pose.orientation
            pos0 = msg.pose.position
            y0, p0, r0 = quat2eulers(quat0.w, quat0.x, quat0.y, quat0.z)
            pos0 = np.array([pos0.x, pos0.y, pos0.z])
            q_calib = quaternion_from_euler(0, 0, -y0)
            # ypr = 
            print(f"Will use {t0} as start yaw0 {y0} pos0 {pos0} qcalib {q_calib}")
            return pos0, q_calib, y0
    return None, None, None

def compress_image_msg(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # cv_image = (img16/256).astype('uint8')
    comp_img = CompressedImage()
    comp_img.header = msg.header
    comp_img.format = "mono8; jpeg compressed"
    succ, _data = cv.imencode(".jpg", cv_image, encode_param)
    comp_img.data = _data.flatten().tolist()
    return comp_img, cv_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('bags', metavar='bags', type=str, nargs='+',
                    help='bags to be synchronized')
    parser.add_argument('-c', '--comp', action='store_true', help='compress the image topics')
    parser.add_argument('-q', '--quality', type=int, default=90, help='quality of the compressed image')
    parser.add_argument('-s', '--show', action='store_true', help="show while sync")
    parser.add_argument('-r', '--realsense', action='store_true', help="is realsense not TUM")
    parser.add_argument('-o', '--output', default="", type=str, help='output path')
    parser.add_argument('-p', '--sync-path', default="", action='store_true', help='sync by path command')
    parser.add_argument('-t','--start-time', nargs='+', help='<Required> Set flag', required=False, type=float)
    parser.add_argument('-u','--duration', nargs='+', help='<Required> Set flag', required=False, type=float)

    args = parser.parse_args()
    bags = args.bags
    dts = {}
    t0s = {}

    if args.sync_path:
        t_traj_min = 1e10
        t_traj_min_bag = ""
        t_trajs = {}
        for bag in bags:
            t0 = get_time0(bag, is_realsense=args.realsense)
            t_traj = get_traj_command_time(bag)
            t_trajs[bag] = t_traj
            d_traj_after_start = t_traj.to_sec() - t0.to_sec()
            if d_traj_after_start < t_traj_min:
                t_traj_min = d_traj_after_start
                t_traj_min_bag = bag
        t0 = t_trajs[bag] - rospy.Duration(t_traj_min)
        print(f"Bag {bag} start at {t0.to_sec()} traj at {t_traj.to_sec()} diff {t_traj.to_sec() - t0.to_sec()}")
        for bag in bags:
            t0s[bag] = t_trajs[bag] - rospy.Duration(t_traj_min)
            dts[bag] = t0 - t0s[bag]
    else:
        t0 = get_time0(bags[0], is_realsense=args.realsense)

        for i in range(len(bags)):
            bag = bags[i]
            t = t_ = get_time0(bag, is_realsense=args.realsense)
            if len(args.start_time) > 0:
                t = t_ + rospy.Duration(args.start_time[i])
            print(f"Bag {bag} start at {t_.to_sec()}, we will use from {t.to_sec()}")
            dts[bag] = t0 - t
            t0s[bag] = t

    # pos0, q_calib, y0 = get_pose0(bags[0])
    
    import pathlib
    output_path = pathlib.Path(bags[0]).resolve() if args.output == "" else pathlib.Path(args.output)
    print(f"{len(bags)} bags to process. Will write to", output_path)

    bridge = CvBridge()
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), args.quality]
    for i in range(len(bags)):
        bag = bags[i]
        output_bag = generate_bagname(bag, output_path, args.comp)
        print("Write bag to", output_bag)
        _dt = dts[bag]
        with rosbag.Bag(output_bag, 'w', compression="bz2") as outbag:
            from nav_msgs.msg import Path
            path = Path()
            path_arr = []
            c = 0
            for topic, msg, t in rosbag.Bag(bag).read_messages():
                if t < t0s[bag]:
                    continue
                if args.duration is not None and t - t0s[bag] > rospy.Duration(args.duration[i]):
                    break
                if msg._has_header:
                    if msg.header.stamp.to_sec() > 0:
                        msg.header.stamp = msg.header.stamp + _dt
                if msg._type == "sensor_msgs/Image" and args.comp:
                    #compress image
                    # msg.data = msg.data.tobytes()
                    # outbag.write(topic, msg, t + _dt)
                    comp_img, cv_image = compress_image_msg(msg)
                    outbag.write(topic+"/compressed", comp_img, t + _dt )
                    if args.show:
                        cv.imshow(topic, cv_image)
                        cv.waitKey(1)
                    continue
                outbag.write(topic, msg, t + _dt )
                if topic == "/vrpn_client/raw_transform" or topic == "/SwarmNode1/pose" or topic=="/leica/pose/relative":
                    if topic == "/vrpn_client/raw_transform":
                        posestamp = PoseStamped()
                        posestamp.header = msg.header
                        posestamp.header.frame_id = "world"
                        pos = msg.transform.translation
                        pos = np.array([pos.x, pos.y, pos.z])
                        pos = yaw_rotate_vec(-y0, pos - pos0)
                        quat = msg.transform.rotation
                        quat = np.array([quat.w, quat.x, quat.y, quat.z])
                        quat = quaternion_multiply(q_calib, quat)
                    elif topic == "/SwarmNode1/pose" or topic=="/leica/pose/relative":
                        posestamp = PoseStamped()
                        posestamp.header = msg.header
                        posestamp.header.frame_id = "world"
                        pos = msg.pose.position
                        pos = np.array([pos.x, pos.y, pos.z])
                        pos = yaw_rotate_vec(-y0, pos - pos0)
                        quat = msg.pose.orientation
                        quat = np.array([quat.w, quat.x, quat.y, quat.z])
                        quat = quaternion_multiply(q_calib, quat)
                    posestamp.pose.position.x = pos[0]
                    posestamp.pose.position.y = pos[1]
                    posestamp.pose.position.z = pos[2]
                    posestamp.pose.orientation.w = quat[0]
                    posestamp.pose.orientation.x = quat[1]
                    posestamp.pose.orientation.y = quat[2]
                    posestamp.pose.orientation.z = quat[3]
                    path.header = posestamp.header
                    outbag.write("/calib_pose", posestamp, t + _dt )
                    path_arr.append(np.concatenate(([msg.header.stamp.to_sec()], pos, quat)))
                    if c % 10 == 0:
                        path.poses.append(posestamp)
                        outbag.write("/calib_path", path, t + _dt )
                    c+=1
            np.savetxt(generate_groundtruthname(bag), path_arr)

