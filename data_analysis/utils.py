import numpy as np
from transformations import * 
from math import *

def quat2eulers(w, x, y ,z):
    r = atan2(2 * (w * x + y * z),
                    1 - 2 * (x * x + y * y))
    p = asin(2 * (w * y - z * x))
    y = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return y, p, r

def quat2eulers_arr(quat):
    ret = quat2eulers(*quat)
    return np.array(ret)

def yaw_rotate_vec(yaw, vec):
    Re = rotation_matrix(yaw, [0, 0, 1])[0:3, 0:3]
    return np.transpose(np.dot(Re, np.transpose(vec)))

def wrap_pi(data):
    return (data + np.pi) % (2 * np.pi) - np.pi

def RMSE(predictions, targets):
    if len(predictions) == 0:
        print("RMSE: no predictions")
        return 0
    err_sq = (predictions-targets)**2
    ret = np.sqrt(np.mean(err_sq))
    # print(predictions, targets, ret)
    ret = np.nan_to_num(ret, 0)

    return ret

def angular_error_ypr_array(ypr1_array, ypr2_array):
    ret = []
    for i in range(len(ypr1_array)):
        ret.append(angular_error_ypr(ypr1_array[i], ypr2_array[i]))
    return np.array(ret)

def angular_error_ypr(ypr1, ypr2):
    quat1 = quaternion_from_euler(ypr1[2], ypr1[1], ypr1[0])
    quat2 = quaternion_from_euler(ypr2[2], ypr2[1], ypr2[0])
    ret = angular_error_quat(quat1, quat2)

    return ret

def angular_error_quat(quat1, quat2):
    dq = quaternion_multiply(quaternion_inverse(quat1), quat2)
    if dq[0] < 0:
        dq = - dq
    angle = 2*acos(dq[0])
    return angle

