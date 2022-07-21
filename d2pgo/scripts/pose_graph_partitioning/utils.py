import matplotlib.pyplot as plt
import graphviz
from transformations import *
from matplotlib.lines import Line2D
import numpy as np
from math import *

plt.rc("figure", figsize=(10,8))
marker_list = [*Line2D.markers.keys()][0:-4]
marker_num = len(marker_list)

def show_graph(graph):
    with open(graph) as f:
        dot_graph = f.read()
        # remove the display(...)
        return graphviz.Source(dot_graph)

# static const size_t keyBits = sizeof(Key) * 8;
# static const size_t chrBits = sizeof(unsigned char) * 8;
# static const size_t indexBits = keyBits - chrBits;

keyBits = 8*8
chrBits= 1*8
indexBits = keyBits - chrBits
chrMask = 255 << indexBits
indexMask = ~chrMask
def convert_keyframe_id(agent_id, _id):
    if _id >= 97 << indexBits:
        #We need to cvt back first
        _id = _id & indexMask
    c_ = 97 + agent_id
    return (c_ << indexBits) | _id
def extrack_keyframe_id(key):
    return ((key & chrMask) >> indexBits)-97, key & indexMask

def quaternion_rotate(q, v):
    mat = quaternion_matrix(q)[0:3,0:3]
    v = np.array([v])
    v = v.transpose()
    v = mat @ v
    v = v.transpose()[0]
    return v

def quat2eulers(quat):
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    r = atan2(2 * (w * x + y * z),
                    1 - 2 * (x * x + y * y))
    p = asin(2 * (w * y - z * x))
    y = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return y, p, r