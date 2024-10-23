from scipy.interpolate import interp1d
import numpy as np
from utils import quat2eulers_arr, wrap_pi, yaw_rotate_vec
from transformations import *

class Trajectory:
    def __init__(self, t, pos, quat):
        self.t = t
        self.pos = pos
        self.quat = quat
        self.ypr = np.apply_along_axis(quat2eulers_arr, 1, quat)
        self.interp()

    def interp(self):
        self.pos_func = interp1d(self.t, self.pos, axis=0,bounds_error=False,fill_value="extrapolate")
        self.ypr[:,0] = np.unwrap(self.ypr[:,0])
        self.ypr_func = interp1d(self.t, self.ypr, axis=0,bounds_error=False,fill_value="extrapolate")
        self.ypr[:,0] = wrap_pi(self.ypr[:,0])

        # Compute velocity
        dp = np.diff(self.pos, axis=0)
        dt = np.diff(self.t)
        self.vel = np.concatenate([np.zeros((1,3)), dp/dt[:,None]], axis=0)
        # Smooth velocity with a moving average filter
        self.vel = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5)/5, mode='same'), 0, self.vel)
        self.vel_func = interp1d(self.t, self.vel, axis=0,bounds_error=False,fill_value="extrapolate")
        
    def length(self, t=10000000):
        mask = self.t < t
        dp = np.diff(self.pos[mask], axis=0)
        length = np.sum(np.linalg.norm(dp,axis=1))
        return length

    def recompute_ypr(self):
        self.ypr = np.apply_along_axis(quat2eulers_arr, 1, self.quat)
        self.interp()
    
    def resample_ypr(self, t):
        ypr = self.ypr_func(t)
        ypr[:,0] = wrap_pi(ypr[:,0])
        return ypr
    
    def resample_pos(self, t):
        return self.pos_func(t)

def ATE_POS(predictions, targets):
    err = predictions-targets
    norm2 = err[:,0]*err[:,0]+err[:,1]*err[:,1]+err[:,2]*err[:,2]
    if np.isnan(norm2).any():
        print("ATE_POS has nan")

    return np.sqrt(np.mean(norm2))

def AVG_DIS(predictions, targets):
    err = predictions-targets
    norm2 = err[:,0]*err[:,0]+err[:,1]*err[:,1]+err[:,2]*err[:,2]
    if np.isnan(norm2).any():
        print("ATE_POS has nan")

    return np.mean(np.sqrt(norm2))

def odometry_covariance_per_meter_with_rp(pos_vo, yaw_vo, pos_gt, yaw_gt, rp_length=1.0, gt_outlier_thres=0.1, show=False,step=1):
    i, j, c = 0, 0, 0
    sqr_err_pos_per_meter = np.zeros((3, 3))
    sqr_err_yaw_per_meter = 0
    ticks = []
    rp_errors = []
    dp_vos = []
    dp_gts = []

    if show:
        import matplotlib.pyplot as plt
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
    if c > 0:
        return sqr_err_pos_per_meter/c, sqr_err_yaw_per_meter/c
    else:
        return np.NaN, np.NaN

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
        import matplotlib.pyplot as plt
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

def find_common_times(times_a, times_b, dt=0.005):
    from sklearn.neighbors import NearestNeighbors
    # times_a_near = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(times_a.reshape(-1, 1))
    times_b_near = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(times_b.reshape(-1, 1))
    distances, indices = times_b_near.kneighbors(times_a.reshape(-1, 1))
    mask = distances.reshape(-1) < dt
    times_a = times_a[mask]
    # distances, indices = times_a_near.kneighbors(times_b.reshape(-1, 1))
    # mask = distances.reshape(-1) < dt
    # times_b = times_b[mask]
    # plt.plot(times_a, marker="+", linestyle="None")
    # plt.plot(times_b, marker=".", linestyle="None")
    return times_a

def align_paths(paths, paths_gt, align_by_first=False, align_with_minize=False, align_coor_only=True):
    # align the first pose in each path to paths_gt
    dpos = None
    for i in paths:
        path = paths[i]
        path_gt = paths_gt[i]
        if dpos is None or not align_by_first:
            if align_with_minize:
                dpos, dyaw, d_att = align_path_by_minimize(path, path_gt, align_coor_only=align_coor_only)
            else:
                t0 = find_common_times(path.t, path_gt.t)[0]
                dpos = path_gt.pos_func(t0) - path.pos_func(t0)
                dyaw = wrap_pi(path_gt.ypr_func(t0)[0] - path.ypr_func(t0)[0])
                dpitch = wrap_pi(path_gt.ypr_func(t0)[1] - path.ypr_func(t0)[1])
                droll = wrap_pi(path_gt.ypr_func(t0)[2] - path.ypr_func(t0)[2])
                d_att = quaternion_from_euler(droll, dpitch, dyaw)
        else:
            t0 = find_common_times(path.t, path_gt.t)[0]
            dpitch = wrap_pi(path_gt.ypr_func(t0)[1] - path.ypr_func(t0)[1])
            droll = wrap_pi(path_gt.ypr_func(t0)[2] - path.ypr_func(t0)[2])
            d_att = quaternion_from_euler(droll, dpitch, dyaw)
        path.pos = yaw_rotate_vec(dyaw, path.pos) + dpos
        att_new = np.apply_along_axis(lambda x: quaternion_multiply(d_att, x), 1, path.quat)
        path.quat = att_new
        path.recompute_ypr()
    return paths

def align_path_by_minimize(path, path_gt, inplace=False, align_coor_only=False):
    from scipy.optimize import minimize
    t = find_common_times(path.t, path_gt.t)
    pos = path.pos_func(t)
    pos_gt = path_gt.pos_func(t)
    ypr = path.ypr_func(t)
    ypr_gt = path_gt.ypr_func(t)
    if align_coor_only:
        def cost(x):
            dpos = x[:3]
            dyaw = x[3]
            pos_err = np.linalg.norm(pos_gt - yaw_rotate_vec(dyaw, pos) - dpos, axis=1)
            yaw_err = np.abs(wrap_pi(ypr_gt[:,0] - ypr[:,0] - dyaw))
            return np.sum(pos_err) #+ np.sum(yaw_err)
        inital_pos = pos_gt[0] - pos[0]
        inital_yaw = wrap_pi(ypr_gt[0, 0] - ypr[0, 0])
        inital_guess = np.concatenate([inital_pos, [inital_yaw]])
        # print("Initial cost", cost(inital_guess))
        res = minimize(cost, inital_guess)
        # print("Optimized:", res)
        relative_pos = res.x[:3]
        relative_yaw = res.x[3]
        relative_pitch = 0
        relative_roll = 0
        if inplace:
            path.pos = yaw_rotate_vec(relative_yaw, path.pos) + relative_pos
            path.ypr = path.ypr + np.array([relative_yaw, 0, 0])
            path.ypr[:, 0] = wrap_pi(path.ypr[:, 0])
            path.interp()
    else:
        def cost(x):
            dpos = x[:3]
            dyaw = x[3]
            d_rp = x[4:5]
            pos_err = np.linalg.norm(pos_gt - yaw_rotate_vec(dyaw, pos) - dpos, axis=1)
            yaw_err = np.abs(wrap_pi(ypr_gt[:,0] - ypr[:,0] - dyaw))
            rp_err = np.abs(wrap_pi(ypr_gt[:,1:] - ypr[:,1:] - d_rp))
            return np.sum(pos_err) #+ np.sum(yaw_err) + np.sum(rp_err)
        inital_pos = pos_gt[0] - pos[0]
        inital_yaw = wrap_pi(ypr_gt[0, 0] - ypr[0, 0])
        inital_guess = np.concatenate([inital_pos, [inital_yaw, 0, 0]])
        # print("Initial cost", cost(inital_guess))
        res = minimize(cost, inital_guess)
        relative_pos = res.x[:3]
        relative_yaw = res.x[3]
        relative_pitch = res.x[4]
        relative_roll = res.x[5]
        if inplace:
            path.pos = yaw_rotate_vec(relative_yaw, path.pos) + relative_pos
            path.ypr = path.ypr + res.x[3:]
            path.ypr[:, 0] = wrap_pi(path.ypr[:, 0])
            path.interp()
    datt = quaternion_from_euler(relative_roll, relative_pitch, relative_yaw)
    return relative_pos, relative_yaw, datt
            