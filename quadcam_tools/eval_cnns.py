#!/usr/bin/env python3
#This file is to evaluate the superpoint+mobilenetvlad
import time
import argparse
import numpy as np
import onnxruntime
import cv2 as cv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the superpoint+mobilenetvlad')
    parser.add_argument('--model', type=str, help='path to the model', default="../models/mobilenetvlad_240x320.onnx")
    parser.add_argument('--input', type=str, help='path to the input', default="../sample_data/fisheye.jpg")
    parser.add_argument('--enable-int8', action='store_true', help='enable int8')
    parser.add_argument('--enable-fp16', action='store_true', help='enable fp16', default="True")
    parser.add_argument('--enable-dla', action='store_true', help='enable dla')
    parser.add_argument('--enable-tensorrt', action='store_true', help='enable tensorrt')
    parser.add_argument('--engine-cache', type=str, help='path to cache of engine', default="")
    parser.add_argument('--calib-table', type=str, help='path to calibration table', default="calibration.flatbuffers")
    parser.add_argument('--width', type=int, default=320, help='input width')
    parser.add_argument('--height', type=int, default=240, help='input height')
    parser.add_argument('--num-run', type=int, default=10, help='num of benchmark runs')
    parser.add_argument('--batch', type=int, default=1, help='num of benchmark runs')
    parser.add_argument('--data-bchw', action='store_true', help='data as batch-channels-height-width')
    args = parser.parse_args()
    print(f'Loading model from {args.model}, input from {args.input}')
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
    sess = onnxruntime.InferenceSession(args.model, providers=providers)
    for input in sess.get_inputs():
        print(f'input name: {input.name}, shape: {input.shape}')
    for output in sess.get_outputs():
        print(f'output name: {output.name}, shape: {output.shape}')
    # Load the input image
    img = cv.imread(args.input, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (args.width, args.height))
    img = img.astype(np.float32)
    if args.data_bchw:
        img = img[np.newaxis,np.newaxis,:,:]
    else:
        img = img[np.newaxis,:,:,np.newaxis]
    if args.batch > 1:
        img = np.repeat(img, args.batch, axis=0)
    print("input shape: ", img.shape)
    # Run the model
    outputs = sess.run(None, {sess.get_inputs()[0].name: img})
    start = time.time()
    for i in range(args.num_run):
        outputs = sess.run(None, {sess.get_inputs()[0].name: img})
    end = time.time()
    print(f'Averged inference time ({args.num_run} runs): {(end - start)*1000/args.num_run:.2f}ms')