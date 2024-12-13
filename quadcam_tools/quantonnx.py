#!/usr/bin/env python3
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table, CalibrationMethod
import os
import numpy as np
import cv2 as cv
import tqdm

def prepare_input(left_img, input_width, input_height, data_bchw=False, multiplier=1.0):
    left_img = cv.resize(left_img, (input_width, input_height))
    if data_bchw:
        return left_img[np.newaxis,np.newaxis,:,:].astype(np.float32)/multiplier
    else:
        return left_img[np.newaxis,:,:,np.newaxis].astype(np.float32)/multiplier

def preprocess_image(image_path, height, width, data_bchw=False, multiplier=1.0):
    img_l = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    data = prepare_input(img_l, width, height, data_bchw, multiplier)
    return data
    
def preprocess_func(image_names, height, width, size_limit=0, data_bchw=False, multiplier=1.0):
    unconcatenated_batch_data = []
    for image_filepath_left in image_names:
        image_data = preprocess_image(image_filepath_left, height, width, data_bchw, multiplier)
        unconcatenated_batch_data.append(image_data)
    return unconcatenated_batch_data

def get_image_names(images_folder):
    names = []
    image_names = os.listdir(images_folder)
    for image_name in image_names:
        image_filepath_left = images_folder + '/' + image_name
        names.append(image_filepath_left)
    return names

class ImageDataReader(CalibrationDataReader):
    def __init__(self, image_names, input_name="image:0", data_bchw=False, width=320, height=240, multiplier=1.0):
        self.image_names = image_names
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.image_height = height
        self.image_width = width
        self.input_name = input_name
        self.data_bchw = data_bchw
        self.multiplier = multiplier

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_names, self.image_height, self.image_width, 
                    size_limit=0, data_bchw=self.data_bchw, multiplier=self.multiplier)
            self.enum_data_dicts = iter([{self.input_name: nhwc_data}  for nhwc_data in nhwc_data_list ])
        return next(self.enum_data_dicts, None)

if __name__ == "__main__":
    calibration_data_folder = "/home/xuhao/output/stereo_calib"
    image_names = get_image_names(calibration_data_folder)
    print(f"Num samples: {len(image_names)} for calib")
    stride = 100

    width = 400
    height = 200
    model_fp32 = "../models/superpoint_series/superpoint_v1_dyn_size.onnx"
    augmented_model_path = f"../models/superpoint_v1_{width}x{height}_augmented_model.onnx"
    print(f"Quantization for TensorRT of superpoint {width}x{height}...")
    calibrator = create_calibrator(model_fp32, [], augmented_model_path=augmented_model_path, 
            calibrate_method = CalibrationMethod.Entropy)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])      
    input_name = "image"
    pbar = tqdm.tqdm(total=len(image_names), colour="green")
    for i in range(0, len(image_names), stride):
        dr = ImageDataReader(image_names[i:i+stride], width=width, height=height, 
                input_name=input_name, data_bchw=True, multiplier=255.0)
        calibrator.collect_data(dr)
        pbar.update(stride)
    write_calibration_table(calibrator.compute_range())
    os.rename("calibration.flatbuffers", "../models/superpoint_calibration.flatbuffers")
    os.rename("calibration.cache", "../models/superpoint_calibration.cache")
    os.rename("calibration.json", "../models/superpoint_calibration.json")

    print(f"Quantization for TensorRT of MobileNetVLAD {width}x{height}...")
    model_fp32 = "../models/netvlad_series/mobilenetvlad_dyn_size.onnx"
    augmented_model_path = f"../models/mobilenetvlad_{width}x{height}_augmented_model.onnx"
    calibrator = create_calibrator(model_fp32, [], augmented_model_path=augmented_model_path, 
            calibrate_method = CalibrationMethod.Entropy)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])      
    input_name = "image:0"
    pbar = tqdm.tqdm(total=len(image_names), colour="green")
    for i in range(0, len(image_names), stride):
        dr = ImageDataReader(image_names[i:i+stride], width=width, height=height, input_name=input_name, data_bchw=False)
        calibrator.collect_data(dr)
        pbar.update(stride)
    write_calibration_table(calibrator.compute_range())
    os.rename("calibration.flatbuffers", "../models/mobilenetvlad_calibration.flatbuffers")
    os.rename("calibration.cache", "../models/mobilenetvlad_calibration.cache")
    os.rename("calibration.json", "../models/mobilenetvlad_calibration.json")
    