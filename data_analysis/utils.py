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

def ATE_POS(predictions, targets):
    err = predictions-targets
    norm2 = err[:,0]*err[:,0]+err[:,1]*err[:,1]+err[:,2]*err[:,2]
    if np.isnan(norm2).any():
        print("ATE_POS has nan")

    return np.sqrt(np.mean(norm2))


def odometry_covariance_per_meter_with_rp(pos_vo, yaw_vo, pos_gt, yaw_gt, rp_length=1.0, gt_outlier_thres=0.1, show=False,step=1):
    i, j, c = 0, 0, 0
    sqr_err_pos_per_meter = np.zeros((3, 3))
    sqr_err_yaw_per_meter = 0
    ticks = []
    rp_errors = []
    dp_vos = []
    dp_gts = []

    if show:
        plt.figure()
        plt.title("rp_errors")
    while i < len(pos_vo) and j < len(pos_vo):
        len_ij = 0
        pos_last = pos_vo[i]
        j = i

        while j < len(pos_vo) - 1 and len_ij < rp_length:
            len_ij += np.linalg.norm(pos_vo[j] - pos_last)
            pos_last = pos_vo[j]
            j += 1
            # if i == 800:
                # print("len_ij", len_ij)

        #Now len ij is approximately rp_length, we compute error of ij
        pos_vo_i = pos_vo[i]
        pos_vo_j = pos_vo[j]
        yaw_vo_i = yaw_vo[i]
        yaw_vo_j = yaw_vo[j]

        dyaw_vo = wrap_pi(yaw_vo_j - yaw_vo_i)
        dpos_vo = yaw_rotate_vec(-yaw_vo_i, pos_vo_j - pos_vo_i)

        pos_gt_i = pos_gt[i]
        pos_gt_j = pos_gt[j]
        yaw_gt_i = yaw_gt[i]
        yaw_gt_j = yaw_gt[j]
        dyaw_gt = wrap_pi(yaw_gt_j - yaw_gt_i)
        dpos_gt = yaw_rotate_vec(-yaw_gt_i, pos_gt_j - pos_gt_i)
        dp_vos.append(dpos_vo)
        dp_gts.append(dpos_gt)
        ticks.append(i)
        
        err = np.transpose(np.array([(dpos_vo - dpos_gt)]))

        if len_ij > 0.01:
            sqr_err_pos = np.matmul(err, np.transpose(err))/len_ij
            sqr_err_yaw = ((dyaw_vo-dyaw_gt))*((dyaw_vo-dyaw_gt))/len_ij
            if np.linalg.norm(sqr_err_pos) < gt_outlier_thres*rp_length:
                sqr_err_pos_per_meter += sqr_err_pos
                sqr_err_yaw_per_meter += sqr_err_yaw
                c += 1
                rp_errors.append(np.linalg.norm(sqr_err_pos))
        i += step

    if show:
        dp_vos = np.array(dp_vos)
        dp_gts = np.array(dp_gts)
        # plt.subplot(311)
        plt.plot(ticks, dp_vos[:,0], label="VO X")
        plt.plot(ticks, dp_gts[:,0], label="GT X")
        plt.plot(ticks, dp_vos[:,0] - dp_gts[:,0], label="ERR X")
        plt.grid()
        plt.legend()
        # plt.subplot(312)
        # plt.plot(ticks, dp_vos[:,1], label="VO Y")
        # plt.plot(ticks, dp_gts[:,1], label="GT Y")
        # plt.grid()
        # plt.legend()
        # plt.subplot(313)
        # plt.plot(ticks, dp_vos[:,2], label="VO Z")
        # plt.plot(ticks, dp_gts[:,2], label="GT Z")
        # plt.grid()
        # plt.legend()
        # plt.grid()
        # print("RP Length", rp_length)
        # plt.plot(rp_errors)
        # plt.grid()
        plt.show()
    return sqr_err_pos_per_meter/c, sqr_err_yaw_per_meter/c

def odometry_covariance_per_meter(pos_vo, yaw_vo, pos_gt, yaw_gt, rp_lengths=[1.0], gt_outlier_thres=1.0, show=False,step=100):
    pos_covs = []
    sum_pos_cov = np.zeros((3, 3))
    sum_yaw_cov = 0
    for rp in rp_lengths:
        pos_cov, yaw_cov = odometry_covariance_per_meter_with_rp(pos_vo, yaw_vo, pos_gt, yaw_gt, rp_length=rp, gt_outlier_thres=gt_outlier_thres, show=show, step=step)
        sum_pos_cov += pos_cov
        sum_yaw_cov += yaw_cov
        pos_covs.append(np.linalg.norm(pos_cov))
    if show:
        plt.figure()
        plt.title("RP vs cov")
        plt.plot(rp_lengths, pos_covs)
        plt.grid()
        plt.show()
    return sum_pos_cov/len(rp_lengths), sum_yaw_cov/len(rp_lengths)

def short_loop_id(id):
    if id < 1e7:
        return f"D{id}"
    return f"L{id //10000 + id%10000}"