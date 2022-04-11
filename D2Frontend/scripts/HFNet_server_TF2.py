#!/usr/bin/env python3
# Tensorflow
from __future__ import print_function

import rospy
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import tensorflow_addons as tfa
from pathlib import Path
import cv2
import numpy as np
from swarm_loop.srv import HFNetSrv, HFNetSrvResponse
from geometry_msgs.msg import Point32
import time
import sys

tfa.register.register_all()

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
    def __init__(self, model_path):
        self.func = self.load(model_path)

    def load(self, output_saved_model_dir):
        saved_model_loaded = tf.saved_model.load(
            output_saved_model_dir, tags=[tag_constants.SERVING])

        frozen_func = graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        return frozen_func
    
    def inference(self, img, k, radius, netvlad_mode = False):
        _img = np.expand_dims(img, axis=2)
        _img = np.array([_img]).astype(np.float)
        _img = tf.convert_to_tensor(_img, dtype=tf.float32)
        start_time = time.time()
        if netvlad_mode:
            output = self.func(image=_img)
        else:
            output = self.func(image=_img, 
                k=tf.convert_to_tensor(k, dtype=tf.int32),
                radius=tf.convert_to_tensor(radius, dtype=tf.int32))

        print( f'Inference hfnet {img.shape} in {( 1000. *(time.time() - start_time) ) }fms')

        return output


class HFNetServer:
    def __init__(self, model_path, k, radius, superpoint_mode = False, netvlad_mode=False):
        self.hfnet = HFNet(model_path)

        tmp_zer = np.random.randn(208, 400)
        self.superpoint_mode = superpoint_mode
        self.netvlad_mode = netvlad_mode
        self.k = k
        self.radius = radius
        self.inference_network_on_image(tmp_zer)

        if superpoint_mode:
            print("SuperPoint Ready")
        elif netvlad_mode:
            print("NetVLAD Ready")
        else:
            print("NFNet ready")
        
    
    def inference_network_on_image(self, img):
        ret = self.hfnet.inference(img, self.k, self.radius, self.netvlad_mode)
        if self.superpoint_mode:
            return ret["keypoints"][0].numpy(), ret["local_descriptors"][0].numpy()
        elif self.netvlad_mode:
            return ret["global_descriptor"][0].numpy()
        else:
            return ret["global_descriptor"][0].numpy(), ret["keypoints"][0].numpy(), ret["local_descriptors"][0].numpy()
    
    def handle_req(self, req):
        start_time = time.time()

        cv_image = imgmsg_to_cv2( req.image )
        if self.superpoint_mode:
            kpts, kp_descs = self.inference_network_on_image(cv_image)
        else:
            global_desc, kpts, kp_descs = self.inference_network_on_image(cv_image)

        ret = HFNetSrvResponse()

        if not self.superpoint_mode:
            ret.global_desc = global_desc
        _kpts = []
        for pt in kpts:
            kp = Point32(pt[0], pt[1], 0)
            _kpts.append(kp)
        ret.keypoints = _kpts
        print(kp_descs.shape)
        ret.local_descriptors = kp_descs.flatten()

        print( 'HFNet return req in %4.4f ms' %( 1000. *(time.time() - start_time) ) )
        return ret

def set_memory_limit(memory_limit):   
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


if __name__ == "__main__":

    superpoint_mode = False
    netvlad_mode = False
    print(sys.argv)
    if len(sys.argv) > 1 and sys.argv[1] == "superpoint":
        superpoint_mode = True
        print("Initializing SuperPoint with tensorflow {}".format(tf.__version__))
    elif len(sys.argv) > 1 and sys.argv[1] == "netvlad":
        netvlad_mode = True
        print("Initializing netvlad with tensorflow {}".format(tf.__version__))



    if superpoint_mode:
        rospy.init_node( 'superpoint_server' )
    else:
        rospy.init_node( 'hfnet_server' )

    model_path = rospy.get_param('~model_path')
    radius = rospy.get_param('~nms_radius')
    k = rospy.get_param('~num_keypoints')
    memory_limit = rospy.get_param('~memory_limit')
    
    set_memory_limit(memory_limit)
    
    hfserver = HFNetServer(model_path, k, radius, superpoint_mode=superpoint_mode, netvlad_mode=netvlad_mode)
    if superpoint_mode:
        s = rospy.Service( '/swarm_loop/superpoint', HFNetSrv, hfserver.handle_req)
    elif netvlad_mode:
        s = rospy.Service( '/swarm_loop/netvlad_mode', HFNetSrv, hfserver.handle_req)
    else:
        s = rospy.Service( '/swarm_loop/hfnet', HFNetSrv, hfserver.handle_req)
    rospy.spin()