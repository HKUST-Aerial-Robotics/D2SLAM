#!/usr/bin/env python3
#This file is to evaluate the depth generation net work
# usage

# python3 ./evaluate_depth_generation.py  --model (cre/hit)  --input [path to ros_bag] --enble-int8  ture --enable-fp16 ture
import time
import argparse
import numpy as np
import onnxruntime
import cv2 as cv
import sys

#modify the path to your own path
input_data_shape = (240,320)
HITNET_PATH = '/root/swarm_ws/src/ONNX-HITNET-Stereo-Depth-estimation/'
HITNET_TYPE = "eth3d" # "eth3d" or "kitti"
HITNET_OPT = True # True for optimized, False for original 



CRENET_PATH = '/root/swarm_ws/src/ONNX-CREStereo-Depth-Estimation'
CRENET_ITER = 5            # Lower iterations are faster, but will lower detail. 
CRENET_VER = "combined" # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.

# left_images_path = ["/root/swarm_ws/src/D2SLAM/sample_data/stereo_pinhole/left_0.png",
#               "/root/swarm_ws/src/D2SLAM/sample_data/stereo_pinhole/left_1.png"]

# right_images_path = ["/root/swarm_ws/src/D2SLAM/sample_data/stereo_pinhole/right_0.png",
#               "/root/swarm_ws/src/D2SLAM/sample_data/stereo_pinhole/right_1.png"]

left_images_path = ["/home/khalil/workspace/d2slam_ws/src/D2SLAM/sample_data/stereo_trainning_data/cam_0_1_compressed/0.png"
             ]

right_images_path = ["/home/khalil/workspace/d2slam_ws/src/D2SLAM/sample_data/stereo_trainning_data/cam_1_0_compressed/0.png",]

def LoadHitNet(): 
    print("Loading hitnet...")
    sys.path.insert(0, HITNET_PATH)
    from hitnet import HitNet, ModelType
    if HITNET_OPT == True:
      model_path = HITNET_PATH + f"/models/eth3d/saved_model_{input_data_shape[0]}x{input_data_shape[1]}/model_float32_opt.onnx"
    else:
      model_path = HITNET_PATH + f"/models/eth3d/saved_model_{input_data_shape[0]}x{input_data_shape[1]}/model_float32.onnx"
    return model_path


def LoadCreNet():
    print("Loading hitnet...")
    sys.path.insert(0, CRENET_PATH)
    from crestereo import CREStereo 
    model_path = CRENET_PATH+f'/models/crestereo_{CRENET_VER}_iter{CRENET_ITER}_{input_data_shape[0]}x{input_data_shape[1]}.onnx'
    return model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the crestereo or hitnet')
    parser.add_argument('--model', type=str, help='path to the model', default="cre")
    parser.add_argument('--model-path', type=str, help='path to the input image', default="")
    parser.add_argument('--enable-int8', action='store_true', help='enable int8')
    parser.add_argument('--enable-fp16', action='store_true', help='enable fp16')
    parser.add_argument('--enable-dla', action='store_true', help='enable dla')
    parser.add_argument('--enable-tensorrt', action='store_true', help='enable tensorrt')
    parser.add_argument('--enable-rgb', action='store_true', help='enabel rgb image or not')
    parser.add_argument('--engine-cache', type=str, help='path to cache of engine', default="")
    parser.add_argument('--calib-table', type=str, help='path to calibration table', default="calibration.flatbuffers")

    parser.add_argument('--width', type=int, default=320, help='input width')
    parser.add_argument('--height', type=int, default=240, help='input height')
    parser.add_argument('--num-run', type=int, default=10, help='num of benchmark runs')
    parser.add_argument('--batch', type=int, default=1, help='num of benchmark runs')
    parser.add_argument('--data-bchw', action='store_true', help='data as batch-channels-height-width')
   
    args = parser.parse_args()
    
    # if args.model == "cre":
    #   model_path = LoadCreNet()
    #   # model_path = "/root/swarm_ws/src/D2SLAM/models/cretereo_combined_iter5_240x320/crestereo_combined_iter5_240x320_augmented_model.onnx"
    # elif args.model == "hit":
    #   model_path = LoadHitNet()
    model_path = args.model_path


    print(f'Loading model {args.model}')
    print(f'tensorrt {args.enable_tensorrt} enable_fp16 {args.enable_fp16} enable_int8 {args.enable_int8}, enable_dla {args.enable_dla}')
    # Load the model
    providers = []
    if args.enable_tensorrt:
        providers.append(('TensorrtExecutionProvider', {
        	'trt_fp16_enable': args.enable_fp16,
			'trt_engine_cache_enable': True,
			'trt_dla_enable': args.enable_dla,
			'trt_int8_enable': args.enable_int8,
			'trt_int8_calibration_table_name': args.calib_table
    	}))
    else:
        providers.append('CUDAExecutionProvider')
    sess = onnxruntime.InferenceSession(model_path, providers=providers)
    for input in sess.get_inputs():
        print(f'input name: {input.name}, shape: {input.shape}')
    for output in sess.get_outputs():
        print(f'output name: {output.name}, shape: {output.shape}')
    input_names = [input_.name for input_ in sess.get_inputs()]
    left_imgs = []
    right_imgs = []
    # Load test input image
    for path in left_images_path:
        left_imgs.append(cv.imread((path), cv.IMREAD_COLOR))
    for path in right_images_path:
        right_imgs.append(cv.imread((path), cv.IMREAD_COLOR))

    # prepare data
    



    if args.model == "hit":
        # left_grey_img = cv.cvtColor(left_imgs[0],cv.COLOR_BGR2GRAY)
        # right_grey_img = cv.cvtColor(right_imgs[0],cv.COLOR_BGR2GRAY)
        left_grey_img = cv.resize(left_grey_img, (args.width, args.height))
        right_grey_img = cv.resize(right_grey_img, (args.width, args.height))

        left_grey_img = left_grey_img.astype(np.float32)
        if args.data_bchw:
            left_grey_img = left_grey_img[np.newaxis,np.newaxis,:,:]
        else:
            left_grey_img = left_grey_img[np.newaxis,:,:,np.newaxis]
        if args.batch > 1:
            left_grey_img = np.repeat(left_grey_img, args.batch, axis=0)

        right_grey_img = right_grey_img.astype(np.float32)
        
        if args.data_bchw:
            right_grey_img = right_grey_img[np.newaxis,np.newaxis,:,:]
        else:
            right_grey_img = right_grey_img[np.newaxis,:,:,np.newaxis]
        if args.batch > 1:
            right_grey_img = np.repeat(right_grey_img, args.batch, axis=0)
        
        # input image
        input = np.concatenate((left_grey_img,right_grey_img), axis= 1)

        print("input shape: ", left_grey_img.shape)
        #Run the model
        # outputs = sess.run(None, {sess.get_inputs()[0].name: input})
        outputs = sess.run(None, {sess.get_inputs()[0].name: left_grey_img, sess.get_inputs()[1].name: right_grey_img})
        start = time.time()
        for i in range(args.num_run):
            outputs = sess.run(None, {sess.get_inputs()[0].name: left_grey_img, sess.get_inputs()[1].name: right_grey_img})
        end = time.time()
        output = np.squeeze(outputs)

        cv.imshow("output", output)
        cv.waitKey(0)

        print(f'Averged inference time ({args.num_run} runs): {(end - start)*1000/args.num_run:.2f}ms')
        
    if args.model == "cre":
        half_size = (int(args.width/2), int(args.height/2))
        full_size = (args.width,args.height )
        init_left_img = cv.resize(left_imgs[0],half_size).astype(np.float32)
        init_right_img = cv.resize(right_imgs[0],half_size).astype(np.float32)
        combine_left_img = cv.resize(left_imgs[0],full_size).astype(np.float32)
        combine_right_img = cv.resize(right_imgs[0],full_size).astype(np.float32)

        init_left_img = np.transpose(init_left_img,(2,0,1))[np.newaxis, :]
        init_right_img = np.transpose(init_right_img,(2,0,1))[np.newaxis, :]
        combine_left_img = np.transpose(combine_left_img,(2,0,1))[np.newaxis, :]
        combine_right_img = np.transpose(combine_right_img,(2,0,1))[np.newaxis, :]
        input = {input_names[0]:init_left_img,
                 input_names[1]:init_right_img,
                  input_names[2]:combine_left_img,
                  input_names[3]:combine_right_img}
        
        #     #Run the model
        outputs = sess.run(None, input)
        start = time.time()
        for i in range(args.num_run):
            outputs = sess.run(None,input)
        end = time.time()
        # output = np.squeeze(outputs)

        # cv.imshow("output", output)
        # cv.waitKey(0)

    if args.model == "acv":
        half_size = (int(args.width/2), int(args.height/2))
        full_size = (args.width,args.height )
        combine_left_img = cv.resize(left_imgs[0],full_size).astype(np.float32)
        combine_right_img = cv.resize(right_imgs[0],full_size).astype(np.float32)

        combine_left_img = np.transpose(combine_left_img,(2,0,1))[np.newaxis, :]
        combine_right_img = np.transpose(combine_right_img,(2,0,1))[np.newaxis, :]
        input = {
                  input_names[0]:combine_left_img,
                  input_names[1]:combine_right_img}
        
        #     #Run the model
        outputs = sess.run(None, input)
        start = time.time()
        for i in range(args.num_run):
            outputs = sess.run(None,input)
        end = time.time()
        output_arr = np.array(outputs)
        output_im = np.squeeze(output_arr)
        print(output_arr[0])
        cv_image = cv.fromarray(output_im)
        cv.imshow("output", cv_image)
        cv.waitKey(0)

        print(f'Averged inference time ({args.num_run} runs): {(end - start)*1000/args.num_run:.2f}ms')