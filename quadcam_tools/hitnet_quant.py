from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table, CalibrationMethod
import os
import numpy as np
import cv2 as cv
import argparse
import json
from PIL import Image

LEFT_IMAGE_DIR_PATH = "/media/khalil/ssd_data/data_set/extract_pinhole/cam_0_1_compressed/"
RIGHT_IMAGE_DIR_PATH = "/media/khalil/ssd_data/data_set/extract_pinhole/cam_1_0_compressed/"
# QUAT_INPUT_MODEL = "/root/swarm_ws/src/ONNX-HITNET-Stereo-Depth-estimation/models/eth3d/saved_model_240x320/model_float32_opt.onnx"
QUAT_INPUT_MODEL = "/home/khalil/workspace/d2slam_ws/src/ONNX-CREStereo-Depth-Estimation/models/crestereo_combined_iter5_240x320.onnx"
OUTPUT_AGUMENT_MODEL = "/home/khalil/workspace/d2slam_ws/src/D2SLAM/quadcam_tools/test.onnx" 




def get_image_names(images_folder):
    names = []
    image_names = os.listdir(images_folder)
    for image_name in image_names:
        image_filepath_left = images_folder + '/' + image_name
        names.append(image_filepath_left)
    return names

def hitnet_preprocess_func(left_image_list, right_image_list,start_index, end_index,height, width, multiplier=1.0):
  concatenated_batch_data = []
  iter_number = len(left_image_list) if len(left_image_list) < len(right_image_list) else len(right_image_list) 
  for i in range(iter_number):
      img_l = cv.imread(left_image_list[i], cv.IMREAD_GRAYSCALE)
      img_r = cv.imread(right_image_list[i], cv.IMREAD_GRAYSCALE)
      img_l = cv.resize(img_l, (width, height))
      img_r = cv.resize(img_r, (width, height))
      data = np.concatenate((img_l[np.newaxis,np.newaxis,:,:].astype(np.float32)/multiplier, img_r[np.newaxis,np.newaxis,:,:].astype(np.float32)/multiplier), axis=1)
      print(f"data type {type(data)}")
      concatenated_batch_data.append(data)
  return concatenated_batch_data
  # def letterbox_image(image, size):
  #   '''resize image with unchanged aspect ratio using padding'''
  #   iw, ih = image.size
  #   w, h = size
  #   scale = min(w/iw, h/ih)
  #   nw = int(iw*scale)
  #   nh = int(ih*scale)

  #   image = image.resize((nw,nh), Image.BICUBIC)
  #   new_image = Image.new('L', size, (128))
  #   new_image.paste(image, ((w-nw)//2, (h-nh)//2))
  #   return new_image
  # l_grey_img = []
  # r_grey_img = []
  # for i in range(5):
  #   left_image_filepath = left_image_list[i]
  #   left_rgb_img = Image.open(left_image_filepath)
  #   left_gray_img = left_rgb_img.convert("L")
  #   model_image_size = (height, width)
  #   left_boxed_image = letterbox_image(left_gray_img, tuple(reversed(model_image_size)))
  #   left_image_data = np.array(left_boxed_image, dtype='float32')
  #   left_image_data /= 255.
  #   left_image_data = np.expand_dims(left_image_data,axis=0)
  #   l_grey_img.append(left_image_data)
  # for i in range(5):
  #   left_image_filepath = right_image_list[i]
  #   left_rgb_img = Image.open(left_image_filepath)
  #   left_gray_img = left_rgb_img.convert("L")
  #   model_image_size = (height, width)
  #   left_boxed_image = letterbox_image(left_gray_img, tuple(reversed(model_image_size)))
  #   left_image_data = np.array(left_boxed_image, dtype='float32')
  #   left_image_data /= 255.
  #   left_image_data = np.expand_dims(left_image_data,axis=0)
  #   r_grey_img.append(left_image_data)
  # for i in range(len(l_grey_img)):
  #   data = np.concatenate((l_grey_img[i], r_grey_img[i]),axis=0)
  #   data = np.expand_dims(data,axis=0)
  #   concatenated_batch_data.append(data)
  #   print(f"image shape{data.shape}  {type(data)}")
  # return concatenated_batch_data

class HITNETImageDataReader(CalibrationDataReader):
    def __init__(self, 
                left_image_list, 
                right_image_list,
                start_index,
                end_index,
                data_name="input",
                data_bchw = False, 
                width = 320, 
                height= 240, 
                multiplier = 1.0, 
                stride= 10):
        self.left_image_list = left_image_list
        self.right_image_list = right_image_list
        self.start_index = start_index
        data_set_length = len(left_image_list) if len(left_image_list) <= len(right_image_list) else len(right_image_list)
        self.end_index = end_index if end_index <=data_set_length else data_set_length
        self.preprocess_flag = True
        self.data_name = data_name
        self.data_bchw = data_bchw
        self.multiplier = multiplier
        self.image_height = height
        self.image_width = width
        self.stride = stride = stride if stride >=1 else 1
        self.enum_data_dicts = iter([])

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data
        self.enum_data_dicts = None
        if self.start_index < self.end_index:
            data = self.load_serial()
            self.start_index += self.stride
            self.enum_data_dicts = iter(data)
            print(f"enum_Data_dicts: {type(self.enum_data_dicts)}:{self.enum_data_dicts}")
            return next(self.enum_data_dicts, None)
        else:
            return None
    
    def load_serial(self):
        nchw_data_list = hitnet_preprocess_func(self.left_image_list,self.right_image_list,self.start_index,self.end_index,self.image_height,self.image_width)
        input_name = self.data_name
        print("Start from index %s ..." % (str(self.start_index)))
        data = []
        for i in range(len(nchw_data_list)):
            nchw_data = nchw_data_list[i]
            data.append({input_name: nchw_data})

        print(f"nchw data list len{len(data)}")
        return data
    


class CRENETImageDataReader(CalibrationDataReader):
    def __init__(self, 
                left_image_list, 
                right_image_list,
                start_index,
                end_index,
                data_name="input",
                data_bchw = False, 
                width = 320, 
                height= 240, 
                multiplier = 1.0, 
                stride= 10):
        self.left_image_list = left_image_list
        self.right_image_list = right_image_list
        self.start_index = start_index
        data_set_length = len(left_image_list) if len(left_image_list) <= len(right_image_list) else len(right_image_list)
        self.end_index = end_index if end_index <=data_set_length else data_set_length
        self.preprocess_flag = True
        self.data_name = data_name
        self.data_bchw = data_bchw
        self.multiplier = multiplier
        self.image_height = height
        self.image_width = width
        self.stride = stride = stride if stride >=1 else 1
        self.enum_data_dicts = iter([])

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data
        self.enum_data_dicts = None
        if self.start_index < self.end_index:
            data_list = self.load_serial()
            self.start_index += self.stride
            self.enum_data_dicts = iter(data_list)
            print(f"enum_Data_dicts: {type(self.enum_data_dicts)}:{self.enum_data_dicts}")
            return next(self.enum_data_dicts, None)
        else:
            return None
    
    def load_serial(self):
        f_image_l, f_image_r, h_image_l, h_image_r = self.crenet_preprocess_func(self.left_image_list,self.right_image_list,self.start_index,self.end_index,self.image_height,self.image_width)
        print("Start from index %s ..." % (str(self.start_index)))
        data = []
        for i in range(len(f_image_l)):
            init_l = h_image_l[i]
            init_r = h_image_r[i]
            next_l = f_image_l[i]
            next_r = f_image_r[i]
            data.append({"init_left": init_l,"init_right":init_r,"next_left":next_l, "next_right":next_r})
        print(f"nchw data list len{len(data)}")
        return data
        ## this function should return 

    def crenet_preprocess_func(self,left_image_list, right_image_list,start_index, end_index,height, width, multiplier=1.0):
      full_img_l = []
      full_img_r = []
      half_img_l = []
      half_img_r = []
      iter_number = len(left_image_list) if len(left_image_list) < len(right_image_list) else len(right_image_list) 
      for i in range(2):
          img_l = cv.imread(left_image_list[i], cv.IMREAD_COLOR)
          img_r = cv.imread(right_image_list[i], cv.IMREAD_COLOR)
          img_l = cv.resize(img_l, (width, height)).astype(np.float32)
          img_r = cv.resize(img_r, (width, height)).astype(np.float32)
          img_half_l = cv.resize(img_l, (int(width/2), int(height/2))).astype(np.float32)
          img_half_r = cv.resize(img_r, (int(width/2), int(height/2))).astype(np.float32)
          img_l = np.transpose(img_l,(2,0,1))[np.newaxis, :]
          img_r = np.transpose(img_r,(2,0,1))[np.newaxis, :]
          img_half_l = np.transpose(img_half_l,(2,0,1))[np.newaxis, :]
          img_half_r = np.transpose(img_half_r,(2,0,1))[np.newaxis, :]
          # print(f"[Input data shape] {img_half_r.shape} ")
          full_img_l.append(img_l)
          full_img_r.append(img_r)
          half_img_l.append(img_half_l)
          half_img_r.append(img_half_r)
      return full_img_l, full_img_r ,half_img_l, half_img_r

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "quantization tool for ONNX structure net")
    parser.add_argument('-m', "--model", type=str, help="quantization target model",default=QUAT_INPUT_MODEL)
    parser.add_argument('-w', "--width", type= int, help="width of input",default= 640)
    parser.add_argument("--height", type=int, help="height of input",default= 480)
    args = parser.parse_args()

    print(f"Do HITNET_{args.height}x{args.width} calibration  {args.model}")
    left_image_list  = get_image_names(LEFT_IMAGE_DIR_PATH)
    right_image_list = get_image_names(RIGHT_IMAGE_DIR_PATH)

    print(f"calibrate model {args.model} with {len(right_image_list)} pairs of images")

    calibrator = create_calibrator(QUAT_INPUT_MODEL,None,augmented_model_path=OUTPUT_AGUMENT_MODEL, calibrate_method = CalibrationMethod.Entropy)
    calibrator.set_execution_providers(["CUDAExecutionProvider"])
    input_name = "input"
    # dr = HITNETImageDataReader(left_image_list, 
    #                           right_image_list,
    #                           start_index= 0,
    #                           end_index= 20,
    #                           data_name=input_name, 
    #                           data_bchw=True, 
    #                           width=args.width, 
    #                           height=args.height)
    dr = CRENETImageDataReader(left_image_list, 
                              right_image_list,
                              start_index= 0,
                              end_index= 20,
                              data_name=input_name, 
                              data_bchw=True, 
                              width=args.width, 
                              height=args.height)
    calibrator.collect_data(dr)

    print("start calirate and quatization")
    write_calibration_table(calibrator.compute_range())
    # print(calibrator.compute_range())
    # for()
    os.rename("calibration.flatbuffers", f"../models/HITNET_{args.height}_calibration.flatbuffers")
    os.rename("calibration.cache", f"../models/HITNET_{args.height}_calibration.cache")
    os.rename("calibration.json", f"../models/HITNET_{args.height}_calibration.json")




