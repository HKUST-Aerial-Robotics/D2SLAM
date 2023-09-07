#!/bin/bash
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1 
export KALIBR_FOCAL_LENGTH_INIT_VALUE=251.729889353184
source /catkin_ws/devel/setup.bash && rosrun kalibr kalibr_calibrate_cameras --bag /data/test.bag --target /data/aprilgrid.yaml --models pinhole-radtan pinhole-radtan --approx-sync 0.01 --topics /cam_0_1/compressed /cam_1_0/compressed