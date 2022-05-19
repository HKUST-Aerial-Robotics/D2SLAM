#/usr/bin/env python3
import rosbag
import rospy
import sys
from pathlib import Path

if __name__ == "__main__":
    bags = sys.argv[1:]
    for topic, msg, t in rosbag.Bag(bags[0]).read_messages():
        t0 = t
        break
    dts = {}
    for bag in bags[1:]:
        print("parse bag", bag)
        for topic, msg, t in rosbag.Bag(bag).read_messages():
            print(f"Bag {bag} start at {t.to_sec()}")
            dts[bag] = t0 - t
            break

    print("Will use", t0, "as start")

    print(dts)
    for bag in bags[1:]:
        print(bag)
        p = Path(bag)
        bagname = p.stem + "-sync.bag"
        output_bag = p.parents[0].joinpath(bagname)
        print("Write bag to", output_bag)
        _dt = dts[bag]
        with rosbag.Bag(output_bag, 'w') as outbag:
            for topic, msg, t in rosbag.Bag(bag).read_messages():
                if msg._has_header:
                    msg.header.stamp = msg.header.stamp + _dt
                outbag.write(topic, msg, t + _dt )