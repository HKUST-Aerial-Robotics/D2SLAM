#!/usr/bin/env python3
from onnxruntime.quantization import quantize_dynamic, QuantType
model_fp32 = "/root/swarm_ws/src/ONNX-CREStereo-Depth-Estimation/models/crestereo_combined_iter5_240x320.onnx"
model_quant = "/root/swarm_ws/src/ONNX-CREStereo-Depth-Estimation/models/crestereo_combined_iter5_240x320_quant.onnx"
augmented_model_path = "/root/swarm_ws/src/ONNX-CREStereo-Depth-Estimation/models/crestereo_combined_iter5_240x320_augmented_model.onnx"
# quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, create_calibrator, write_calibration_table, CalibrationMethod
import os
import numpy as np
import cv2 as cv
import re

def hwc2chw(img_input):
    img_input = img_input.transpose(2, 0, 1)
    # img_input = img_input[np.newaxis,:,:,:]    
    return img_input

def prepare_input(left_img, right_img, input_width, input_height):
    left_img = cv.resize(left_img, (input_width, input_height))
    right_img = cv.resize(right_img, (input_width,input_height))
    init_left_img = cv.resize(left_img, (input_width//2, input_height//2))
    init_right_img = cv.resize(right_img, (input_width//2, input_height//2))
    left_img = hwc2chw(left_img)
    right_img = hwc2chw(right_img)
    init_left_img = hwc2chw(init_left_img)
    init_right_img = hwc2chw(init_right_img)
    # left_img = np.expand_dims(left_img,2)
    # right_img = np.expand_dims(right_img,2)
    # init_left_img = np.expand_dims(init_left_img,2)
    # init_right_img = np.expand_dims(init_right_img,2)

    return  np.expand_dims(init_left_img, 0).astype(np.float32), np.expand_dims(init_right_img, 0).astype(np.float32), \
         np.expand_dims(left_img, 0).astype(np.float32), np.expand_dims(right_img, 0).astype(np.float32)

def preprocess_image(image_path, image_path_r, height, width, channels=3):
    print(f"image_path left {image_path}, right {image_path_r}")
    img_l = cv.imread(image_path, cv.IMREAD_COLOR)
    img_r = cv.imread(image_path_r, cv.IMREAD_COLOR)
    data = prepare_input(img_l, img_r, width, height)
    return data
    
def preprocess_func(image_names, height, width, size_limit=0):
    unconcatenated_batch_data = []
    for image_filepath_left, image_filepath_right in image_names:
        image_data = preprocess_image(image_filepath_left, image_filepath_right, height, width)
        unconcatenated_batch_data.append(image_data)
    # batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return unconcatenated_batch_data

# def get_image_names(images_folder):
#     names = []
#     image_names = os.listdir(images_folder)
#     for image_name in image_names:
#         if "left" in image_name:
#             image_filepath_left = images_folder + '/' + image_name
#             image_filepath_right = images_folder + '/' + image_name.replace("left", "right")
#             names.append((image_filepath_left, image_filepath_right))
#     return names

def get_image_names(images_folder):
    names = []
    image_names = os.listdir(images_folder)
    image_namse_sorted = sorted(image_names, key=lambda x: int(re.search(r'\d+', x).group()))
    for image_name in image_namse_sorted:
        # if "left" in image_name:
        image_file_path = images_folder + '/' + image_name
        names.append(image_file_path)
    print(f"sorted_file_names: {names}")
    return names

class StereoDataReader(CalibrationDataReader):
    def __init__(self, image_names):
        self.image_names = image_names
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.image_height = 240
        self.image_width = 320

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            print("self.image_names: ", self.image_names)
            nhwc_data_list = preprocess_func(self.image_names, self.image_height, self.image_width, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'init_left': nhwc_data[0], 'init_right': nhwc_data[1], 'next_left': nhwc_data[2], 'next_right': nhwc_data[3] } for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

# calibration_data_folder = "./trainning_data"
left_image_path = "/data/extract_pinhole/cam_0_1_compressed/"
right_image_path = "/data/extract_pinhole/cam_1_0_compressed/"


# image_names = get_image_names(calibration_data_folder)
left_image_names = get_image_names(left_image_path)
right_image_names = get_image_names(right_image_path)
image_num = len(left_image_names) if len(left_image_names) < len(right_image_names) else len(right_image_names)
print(f"Num samples: {image_num} for calib")
image_names = []
for i in range(len(left_image_names)-1):
    image_names.append((left_image_names[i], right_image_names[i]))

calib_num = len(image_names)
print(f"Num samples: {len(image_names)} for calib")
print("Quantization for TensorRT...")
stride = 5
calibrator = create_calibrator(model_fp32, [], augmented_model_path=augmented_model_path, calibrate_method = CalibrationMethod.Entropy)
calibrator.set_execution_providers(["CPUExecutionProvider"])      
for i in range(0, calib_num, stride):
    print("stride", i)
    dr = StereoDataReader(image_names[i:i+stride])
    calibrator.collect_data(dr)
write_calibration_table(calibrator.compute_range())