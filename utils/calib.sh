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
    --bag /data/quad_cam_calib-split.bag --target /data/aprilgrid.yaml \
    --models omni-radtan   --approx-sync 0 \
    --topics /arducam/image_0/compressed 
