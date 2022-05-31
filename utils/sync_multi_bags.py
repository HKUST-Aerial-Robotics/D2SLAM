#/usr/bin/env python3
import rosbag
import sys
from transformations import *
import numpy as np
from math import *
from geometry_msgs.msg import PoseStamped
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

def generate_bagname(bag):
    from pathlib import Path
    p = Path(bag)
    bagname = p.stem + "-sync-calib.bag"
    output_bag = p.parents[0].joinpath(bagname)
    # output_bag = "/home/xuhao/Dropbox/data/d2slam/tum_datasets/" + bagname
    return output_bag

if __name__ == "__main__":
    bags = sys.argv[1:]
    for topic, msg, t in rosbag.Bag(bags[0]).read_messages():
        t0 = t
        break
    for topic, msg, t in rosbag.Bag(bags[0]).read_messages():
        if topic == "/vrpn_client/raw_transform":
            pose0 = msg
            break
    dts = {}
    for bag in bags[1:]:
        print("parse bag", bag)
        for topic, msg, t in rosbag.Bag(bag).read_messages():
            print(f"Bag {bag} start at {t.to_sec()}")
            dts[bag] = t0 - t
            break
    quat0 = pose0.transform.rotation
    pos0 = pose0.transform.translation
    y0, p0, r0 = quat2eulers(quat0.w, quat0.x, quat0.y, quat0.z)
    pos0 = np.array([pos0.x, pos0.y, pos0.z])
    q_calib = quaternion_from_euler(0, 0, -y0)
    # ypr = 
    print(f"Will use {t0} as start yaw0 {y0} pos0 {pos0} qcalib {q_calib}")

    print(dts)
    for bag in bags:
        output_bag = generate_bagname(bag)
        print("Write bag to", output_bag)
        if bag not in dts:
            _dt = rospy.Duration(0)
        else:
            _dt = dts[bag]
        with rosbag.Bag(output_bag, 'w') as outbag:
            from nav_msgs.msg import Path
            path = Path()
            c = 0
            for topic, msg, t in rosbag.Bag(bag).read_messages():
                if msg._has_header:
                    msg.header.stamp = msg.header.stamp + _dt
                outbag.write(topic, msg, t + _dt )
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
                    posestamp.pose.position.x = pos[0]
                    posestamp.pose.position.y = pos[1]
                    posestamp.pose.position.z = pos[2]
                    posestamp.pose.orientation.w = quat[0]
                    posestamp.pose.orientation.x = quat[1]
                    posestamp.pose.orientation.y = quat[2]
                    posestamp.pose.orientation.z = quat[3]
                    path.header = posestamp.header
                    outbag.write("/calib_pose", posestamp, t + _dt )
                    if c % 10 == 0:
                        path.poses.append(posestamp)
                        outbag.write("/calib_path", path, t + _dt )
                    c+=1

