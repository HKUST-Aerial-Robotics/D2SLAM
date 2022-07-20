#!/usr/bin/env python
import rospy
from dslam_pose_graph_opti.msg import *
from std_msgs.msg import Int32
import sys

class DSLAMCommSimulator(object):
    def __init__(self):
        self.simulate_delay_ms = rospy.get_param("~simulate_delay_ms")
        self.exchange_poses_pub = rospy.Publisher('~exchange_poses_pub', exchange_poses, queue_size=10)
        self.local_poses_pub = rospy.Publisher('~local_poses_pub', exchange_poses, queue_size=10)
        self.exchange_poses_sub = rospy.Subscriber("~exchange_poses_sub", exchange_poses, self.exchange_poses_callback, queue_size=10000, tcp_nodelay=True)
        self.start_solve_trigger_sub = rospy.Subscriber("~start_solve_trigger_sub", Int32, self.start_solve_trigger_callback, queue_size=10, tcp_nodelay=True)
        
        self.exchange_poses_queue = []
        self.exchange_poses_stamp_queue = []
        self.poses_count = 0
        rospy.loginfo("[dslam_simulate_comm] Simulate {}ms latency for comm.".format(self.simulate_delay_ms))

    def exchange_poses_callback(self, exchange_poses):
        self.exchange_poses_queue.append(exchange_poses)
        self.exchange_poses_stamp_queue.append(rospy.get_rostime())
    
    def start_solve_trigger_callback(self, data):
        if data.data == 0:
            rospy.loginfo(f"[dslam_simulate_comm] Total exchange_poses {self.poses_count} last {len(self.exchange_poses_stamp_queue)}")
            rospy.signal_shutdown("[dslam_simulate_comm] Finish")
            sys.exit(0)
    
    def spinOnce(self):
        while not rospy.is_shutdown() and len(self.exchange_poses_stamp_queue) > 0:
            dt = (rospy.get_rostime() - self.exchange_poses_stamp_queue[0]).to_sec()
            # print(dt, self.simulate_delay_ms/1000.0, len(self.exchange_poses_stamp_queue))
            if  dt > self.simulate_delay_ms/1000.0:
                self.exchange_poses_stamp_queue.pop(0)
                exchange_poses = self.exchange_poses_queue.pop(0)
                self.exchange_poses_pub.publish(exchange_poses)
                self.poses_count += len(exchange_poses.poses)
            else:
                break
#Performance is highly limited.... May be we need to switch to c++

if __name__ == '__main__':
    rospy.init_node("dslam_simulate_comm")
    sim = DSLAMCommSimulator()
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        sim.spinOnce()
        rate.sleep()