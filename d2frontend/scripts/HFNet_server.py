#!/usr/bin/env python3
# Tensorflow
from __future__ import print_function

import rospy
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
tf.contrib.resampler
from pathlib import Path
import cv2
import numpy as np
from swarm_loop.srv import HFNetSrv, HFNetSrvResponse
from geometry_msgs.msg import Point32
import time

def imgmsg_to_cv2( msg ):
    assert msg.encoding == "8UC3" or msg.encoding == "8UC1" or msg.encoding == "bgr8" or msg.encoding == "mono8", \
        "Expecting the msg to have encoding as 8UC3 or 8UC1, received"+ str( msg.encoding )
    if msg.encoding == "8UC3" or msg.encoding=='bgr8':
        X = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        return X

    if msg.encoding == "8UC1" or msg.encoding=='mono8':
        X = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return X

class HFNet:
    def __init__(self, model_path, outputs, mem_usage):
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = mem_usage

        self.session = tf.Session(config=config)
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 1))

        net_input = self.image_ph[None]
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n+':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')

    
        
    def inference(self, image, nms_radius=4, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)


class HFNetServer:
    def __init__(self, model_path, mem_usage, num_kpts=200, nms_radius = 4 ):
        outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
        self.hfnet = HFNet(model_path, outputs, mem_usage)
        self.num_kpts = num_kpts
        self.nms_radius = nms_radius

        tmp_zer = np.random.randn(208, 400)
        self.inference_network_on_image(tmp_zer)
        print("NFNet ready")
    
    def inference_network_on_image(self, img):
        img = np.expand_dims(img, axis=2)
        print("Try inference hfnet", img.shape)
        ret = self.hfnet.inference(img, self.nms_radius, self.num_kpts)
        print("Inference hfnet done")
        return ret["global_descriptor"], ret["keypoints"], ret["local_descriptors"]
    
    def handle_req(self, req):
        start_time = time.time()

        cv_image = imgmsg_to_cv2( req.image )
        global_desc, kpts, kp_descs = self.inference_network_on_image(cv_image)
        ret = HFNetSrvResponse()
        ret.global_desc = global_desc
        _kpts = []
        for pt in kpts:
            kp = Point32(pt[0], pt[1], 0)
            _kpts.append(kp)
        ret.keypoints = _kpts
        print(kp_descs.shape)
        ret.local_descriptors = kp_descs.flatten()

        print( 'HFNet return req in %4.4fms' %( 1000. *(time.time() - start_time) ) )
        return ret

def solve_cudnn_error():   
    # return
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    except RuntimeError as e:
        print(e)



if __name__ == "__main__":
    print("Initializing HFNet... with tensorflow {}".format(tf.__version__))
    rospy.init_node( 'hfnet_server' )
    nms_radius = rospy.get_param('~nms_radius')
    num_keypoints = rospy.get_param('~num_keypoints')
    model_path = rospy.get_param('~model_path')
    mem_usage = rospy.get_param('~mem_usage')
    
    # solve_cudnn_error()
    
    hfserver = HFNetServer(model_path, mem_usage, num_keypoints, nms_radius)
    s = rospy.Service( '/swarm_loop/hfnet', HFNetSrv, hfserver.handle_req)
    rospy.spin()