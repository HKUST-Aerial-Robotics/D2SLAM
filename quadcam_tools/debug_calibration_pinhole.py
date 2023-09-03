from generate_stereo_from_bag import kablirCalibratePinholeCMD, kablirCalibratePinhole
import os
topic_a = "/cam_3_1/compressed"
topic_b = "/cam_0_0/compressed"
bagfile = "/media/khalil/ssd_data/data_set/omni_cam_0808/output/stereo_calibration.bag"
output_calib_name = "stereo_calib_0_0_300_600"
verbose = False
init_focal_length = 251.729889353184

if __name__ == "__main__":
  kablirCalibratePinhole(topic_a, topic_b, bagfile, output_calib_name, verbose = verbose, init_focal_length = init_focal_length)