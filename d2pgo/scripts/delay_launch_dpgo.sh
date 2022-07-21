#!/usr/bin/env bash
#source /opt/ros/melodic/setup.bash
sleep 2
echo "Triggering DPGO"
rostopic pub /dpgo/start_solve_trigger std_msgs/Int32 "data: 1" -l1
