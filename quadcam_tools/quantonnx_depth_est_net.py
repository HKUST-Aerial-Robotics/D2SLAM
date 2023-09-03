#!/usr/bin/env python3
from onnxruntime.quantization import quantize_dynamic, QuantType
model_fp32 = "/home/khalil/workspace/ONNX-CREStereo-Depth-Estimation/models/crestereo_combined_iter5_240x320.onnx"
model_quant = "/home/khalil/workspace/ONNX-CREStereo-Depth-Estimation/models/crestereo_combined_iter5_240x320_quant.onnx"
augmented_model_path = "/home/khalil/workspace/ONNX-CREStereo-Depth-Estimation/models/crestereo_combined_iter5_240x320_augmented_model.onnx"
# quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, create_calibrator, write_calibration_table, CalibrationMethod
import os
import numpy as np
import cv2 as cv
import re
import argparse

def hwc2chw(img_input):
    img_input = img_input.transpose(2, 0, 1)
    # img_input = img_input[np.newaxis,:,:,:]    
    return img_input


# function for crestereo net
def cre_prepare_input(left_img, right_img, input_width, input_height):
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

# function for hitnet
def hit_prepare_input(left_img, right_img, input_width, input_height):
    # if 3 channel, convert to 1 channel
    left_img = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right_img = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)
    left_img = cv.resize(left_img, (input_width, input_height))
    right_img = cv.resize(right_img, (input_width,input_height))

    left_img = np.expand_dims(left_img, axis=(0,1)).astype(np.float32)
    right_img = np.expand_dims(right_img, axis=(0,1)).astype(np.float32)
    print(f"[Debug] left_img shape {left_img.shape}")
    # left_img = hwc2chw(left_img)
    # right_img = hwc2chw(right_img)

    return np.concatenate((left_img, right_img), axis=1)


def preprocess_image(image_path, image_path_r, height, width, prepare_function,channels=3):
    print(f"image_path left {image_path}, right {image_path_r}")
    img_l = cv.imread(image_path, cv.IMREAD_COLOR)
    img_r = cv.imread(image_path_r, cv.IMREAD_COLOR)
    data = prepare_function(img_l, img_r, width, height)
    return data
    
def preprocess_func(image_names, height, width, prepare_function,size_limit=0):
    unconcatenated_batch_data = []
    for image_filepath_left, image_filepath_right in image_names:
        image_data = preprocess_image(image_filepath_left, image_filepath_right, height, width, prepare_function)
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
    # print(f"sorted_file_names: {names}")
    return names

class StereoDataReader(CalibrationDataReader):
    def __init__(self, image_names, prepare_func, type):
        self.image_names = image_names
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.image_height = 240
        self.image_width = 320
        self.prepare_func = prepare_func
        self.type = type 

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            print("self.image_names: ", self.image_names)
            nhwc_data_list = preprocess_func(self.image_names, self.image_height, self.image_width, self.prepare_func, size_limit=0)
            self.datasize = len(nhwc_data_list)
            if self.type == "crenet":
                self.enum_data_dicts = iter([{'init_left': nhwc_data[0], 'init_right': nhwc_data[1], 'next_left': nhwc_data[2], 'next_right': nhwc_data[3] } for nhwc_data in nhwc_data_list])
            elif self.type == "hitnet":
                self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


left_image_path = "/media/khalil/ssd_data/data_set/depth_trainning_data/cam_0_1_compressed/"
right_image_path = "/media/khalil/ssd_data/data_set/depth_trainning_data/cam_1_0_compressed/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fp32', type=str, default='./model.onnx', help='Path to the FP32 model')
    parser.add_argument('--augmented_model_path', type=str, default='./model_aug.onnx', help='Path to the augmented model')
    parser.add_argument('--type', type=str, default='crenet', help='Type of the model crenet or hitnet')
    parser.add_argument('--calib_number', type=int, default=10, help='Path to the calibration data folder')
    parser.add_argument('--width', type=int, default=320, help='Network input width')
    parser.add_argument('--height', type=int, default=240, help='Network input height')
    args = parser.parse_args()

    model_fp32 = args.model_fp32
    if not os.path.exists(model_fp32):
        print(f"model_fp32 {model_fp32} does not exist")
        exit(1)
    augmented_model_path = args.augmented_model_path

    if args.type == 'crenet':
        prepare_function = cre_prepare_input
    elif args.type == 'hitnet':
        prepare_function = hit_prepare_input
    else:
        print(f"model type {args.type} is not supported")
        exit(1)

    # image_names = get_image_names(calibration_data_folder)
    left_image_names = get_image_names(left_image_path)
    right_image_names = get_image_names(right_image_path)
    image_num = len(left_image_names) if len(left_image_names) < len(right_image_names) else len(right_image_names)
    print(f"Num samples: {image_num} for calib")
    image_names = []
    for i in range(len(left_image_names)-1):
        image_names.append((left_image_names[i], right_image_names[i]))

    calib_num = len(image_names)
    if args.calib_number < calib_num:
        calib_num = args.calib_number
    
    print(f"Num samples: {len(image_names)} for calib")
    print("Quantization for TensorRT...")
    stride = 10
    calibrator = create_calibrator(model_fp32, [], augmented_model_path=augmented_model_path, calibrate_method = CalibrationMethod.Entropy)
    calibrator.set_execution_providers(["TensorrtExecutionProvider"])      
    for i in range(0, calib_num, stride):
        print("stride", i)
        dr = StereoDataReader(image_names[i:i+stride],prepare_function, args.type)
        calibrator.collect_data(dr)
    output = calibrator.compute_range()
    print("output", output)
    # convert float32 to float thats wierd but it works, json drop can't handle float32
    float_dic = {}
    for key, value in output.items():
        print(f"[Float32]key: {key}, value: {value}")
        float_dic[key] = (float(value[0]), float(value[1]))
        print(f"[Float]key: {key}, value: {float_dic[key]}")
    print("write_calibration_table")
    write_calibration_table(float_dic)
    print("write_calibration_table done")

    flatbuffers_output_path = args.augmented_model_path.replace(".onnx", ".flatbuffers")
    cache_output_path = args.augmented_model_path.replace(".onnx", ".cache")
    calib_json_output_path = args.augmented_model_path.replace(".onnx", ".json")
    os.rename("./calibration.flatbuffers", flatbuffers_output_path)
    os.rename("./calibration.cache", cache_output_path)
    os.rename("./calibration.json", calib_json_output_path)
