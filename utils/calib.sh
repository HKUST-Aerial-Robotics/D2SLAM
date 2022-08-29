source devel/setup.bash
rosrun kalibr kalibr_calibrate_cameras \
    --bag /data/quad_cam_calib-split.bag --target /data/aprilgrid.yaml \
    --models omni-radtan omni-radtan omni-radtan omni-radtan \
    --topics /arducam/image_0/compressed /arducam/image_1/compressed /arducam/image_3/compressed /arducam/image_2/compressed

# rosrun kalibr kalibr_calibrate_cameras \
#     --bag /data/quad_cam_calib-split.bag --target /data/aprilgrid.yaml \
#     --models omni-radtan  --approx-sync 0 \
#     --topics /arducam/image_3/compressed

# source devel/setup.bash
# rosrun kalibr kalibr_calibrate_cameras \
#     --bag /data/quad_cam_calib-split.bag --target /data/aprilgrid.yaml \
#     --models omni-radtan omni-radtan --approx-sync 0 \
#     --topics /arducam/image_0/compressed /arducam/image_1/compressed

rosrun kalibr kalibr_calibrate_cameras \
    --bag /data/quadcam_calib_2-split.bag --target /data/aprilgrid.yaml \
    --models omni-radtan   --approx-sync 0 \
    --topics /arducam/image_3/compressed 


rosrun kalibr kalibr_calibrate_cameras \
    --bag /data/quadcam_calib_2022_8_26_stereos.bag --target /data/aprilgrid.yaml \
    --models pinhole-radtan pinhole-radtan --approx-sync 0.01 \
    --topics /cam_3_1/compressed /cam_2_0/compressed