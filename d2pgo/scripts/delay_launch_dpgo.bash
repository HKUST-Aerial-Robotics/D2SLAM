#!/usr/bin/env bash
source /opt/ros/melodic/setup.bash
sleep 0.2
rostopic pub /swarm/start_solve_trigger std_msgs/Int32 "data: 1" -l1
