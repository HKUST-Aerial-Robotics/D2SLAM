#this script can transfer a vitual stereo calibration file down to specific height and width
#!/usr/bin/env python3
import yaml
import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='virtual stereo calibration file down sample')
  parser.add_argument("-i","--input", type=str, help="input calibration file")
  parser.add_argument("-w","--width", type=int, default=600, help="width of image")
  parser.add_argument("--height", type=int, default=300, help="height of image")
  args = parser.parse_args()
  if args.input == "":
    print("[INPUT ERROR] shoudl provide input calibration file")
    exit(1)
  else:
    with open(args.input, 'r') as stream:
      try:
        calib = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
        print(exc)
  # generate new calibration file name
  output_name = args.input.rsplit(".",1)[0] + f"-ds-{args.width}x{args.height}.yaml"
  print("output_file_path:", output_name)
  #rewrite intrinsic parametes and resolution
  calib["cam0"]["intrinsics"] = [calib["cam0"]["intrinsics"][0]*args.width/calib["cam0"]["resolution"][0], 
                                 calib["cam0"]["intrinsics"][1]*args.height/calib["cam0"]["resolution"][1],
                                  calib["cam0"]["intrinsics"][2]*args.width/calib["cam0"]["resolution"][0],
                                  calib["cam0"]["intrinsics"][3]*args.height/calib["cam0"]["resolution"][1]]
  calib["cam0"]["resolution"] = [args.width, args.height]
  calib["cam1"]["intrinsics"] = [calib["cam1"]["intrinsics"][0]*args.width/calib["cam1"]["resolution"][0],
                                  calib["cam1"]["intrinsics"][1]*args.height/calib["cam1"]["resolution"][1],
                                  calib["cam1"]["intrinsics"][2]*args.width/calib["cam1"]["resolution"][0],
                                  calib["cam1"]["intrinsics"][3]*args.height/calib["cam1"]["resolution"][1]]
  calib["cam1"]["resolution"] = [args.width, args.height]
# write to new calibration file with original format
  with open(output_name, 'w') as f:
    yaml.dump(calib, f, default_flow_style=False)
  print("Done")



