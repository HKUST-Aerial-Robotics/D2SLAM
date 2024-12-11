#!/usr/bin/env python3
from cmath import nan
import matplotlib.pyplot as plt
import numpy as np
from transformations import * 
from numpy.linalg import norm
import scipy.stats as stats
from utils import *
from trajectory import *

def read_path_from_csv(path, t0=None, delimiter=None,dte=None, reset_orientation=False, time_multiplier=1.0):
    arr = np.loadtxt(path, delimiter=delimiter)
    t = arr[:, 0] * time_multiplier
    if t0 is None:
        t0 = t[0]
    t = t - t0
    pos = arr[:, 1:4]
    quat = arr[:, 4:8]
    # Per line normalization
    quat = np.apply_along_axis(lambda x: x/norm(x), 1, quat)
    if dte is not None:
        mask = (t < dte) & (t > 0)
    else:
        mask = t > 0
    t = t[mask]
    pos = pos[mask]
    quat = quat[mask]
    if reset_orientation:
        pos0 = pos[0]
        quat0 = quat[0]
        y0, _, _ = quat2eulers(quat0[0], quat0[1], quat0[2], quat0[3])
        q_calib_pos = quaternion_from_euler(0, 0, -y0)
        q_calib_att = quaternion_inverse(quat0)
        #Apply rotation to all poses
        quat = np.apply_along_axis(lambda x: quaternion_multiply(q_calib_att, x), 1, quat)
        pos = np.apply_along_axis(lambda x: quaternion_matrix(q_calib_pos)[:3,:3]@ (x - pos0), 1, pos)
    return Trajectory(t, pos, quat), t0

def read_paths(folder, nodes, prefix="d2vins_", suffix=".csv", t0=None, dte=None, reset_orientation=False):
    ret = {}
    for drone_id in nodes:
        try:
            ret[drone_id], t0 = read_path_from_csv(f"{folder}/{prefix}{drone_id}{suffix}", t0, dte=dte, reset_orientation=reset_orientation)
        except Exception as e:
            print(f"Failed to read {folder}/{prefix}{drone_id}{suffix}")
    return ret, t0

def read_multi_folder(folder, nodes, enable_pgo=True, t0=None):
    paths = {}
    paths_pgo = {}
    for i in nodes:
        output_folder = folder + str(i) + "/"
        _paths, t0 = read_paths(output_folder, [i], t0=t0)
        if enable_pgo:
            _paths_pgo, t0 = read_paths(output_folder, [i], prefix="pgo_", t0=t0)
            paths_pgo[i] = _paths_pgo[i]
        paths[i] = _paths[i]
    if len(paths_pgo) == 0:
        return paths, None, t0
    return paths, paths_pgo, t0
    
def plot_fused(nodes, poses_fused, poses_gt=None, poses_pgo=None , output_path="/home/xuhao/output/", id_map = None, figsize=(6, 6), plot_each=True, plot_3d = True):
    import matplotlib as mpl
    mpl.style.use('seaborn-whitegrid')

    if id_map is None:
        id_map = {}
        for i in nodes:
            id_map[i] = i

    if plot_3d:
        fig = plt.figure("plot3d", figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        for i in nodes:
            if poses_gt is not None:
                ax.plot(poses_gt[i].pos[:,0], poses_gt[i].pos[:,1],poses_gt[i].pos[:,2], label=f"GT {i}")
            if poses_pgo is not None:
                ax.plot(poses_pgo[i].pos[:,0], poses_pgo[i].pos[:,1],poses_pgo[i].pos[:,2], label=f"PGO {i}")
            ax.plot(poses_fused[i].pos[:,0], poses_fused[i].pos[:,1],poses_fused[i].pos[:,2], label=f"$D^2$VINS {id_map[i]}")
        
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        
        plt.legend()
        plt.savefig(output_path+"plot3d.png")

        #Plot Fused Vs GT 3D
        fig = plt.figure("FusedVsGT3D", figsize=figsize)
        # fig.suptitle("Fused Vs GT 3D")
        k = 0
        for i in nodes:
            _id = id_map[i]
            ax = fig.add_subplot(1, len(nodes), k+1, projection='3d')
            ax.set_title(f"Drone {_id}")
            if poses_gt is not None:
                ax.plot(poses_gt[i].pos[:,0], poses_gt[i].pos[:,1],poses_gt[i].pos[:,2], label=f"Ground Truth")
            ax.plot(poses_fused[i].pos[:,0], poses_fused[i].pos[:,1],poses_fused[i].pos[:,2], label=f"$D^2$VINS")
            if poses_pgo is not None:
                ax.plot(poses_pgo[i].pos[:,0], poses_pgo[i].pos[:,1],poses_pgo[i].pos[:,2], label=f"$D^2$PGO")
            if i == nodes[0]:
                plt.legend()
            ax.set_xlabel('$X$')
            ax.set_ylabel('$Y$')
            ax.set_zlabel('$Z$')
            k += 1
        plt.savefig(output_path+"FusedVsGT3D.pdf")

    fig = plt.figure("Fused Multi 2d", figsize=figsize)
    plt.gca().set_aspect('equal')
    
    for i in nodes:
        _id = id_map[i]
        if poses_pgo is not None:
            plt.plot(poses_pgo[i].pos[:,1], poses_pgo[i].pos[:,0], label=f"$D^2$PGO {_id}")
            plt.plot(poses_fused[i].pos[:,1], poses_fused[i].pos[:,0], label=f"$D^2$VINS {_id}", linestyle='--')
        else:
            plt.plot(poses_fused[i].pos[:,1], poses_fused[i].pos[:,0], label=f"$D^2$VINS {_id}")
        if poses_gt is not None:
            plt.plot(poses_gt[i].pos[:,1], poses_gt[i].pos[:,0], label=f"Ground Truth {_id}")

    plt.ylabel('$X$')
    plt.xlabel('$Y$')
    plt.legend()
    plt.grid()
    plt.savefig(output_path+"fused2d.pdf")
    if not plot_each:
        return
    for i in nodes:
        _id = id_map[i]
        fig = plt.figure(f"Drone {i} fused Vs GT Pos", figsize=figsize)
        #fig.suptitle(f"Drone {i} fused Vs GT 1D")
        ax1, ax2, ax3 = fig.subplots(3, 1)

        t_ = poses_fused[i].t
        if poses_gt is not None:
            ax1.plot(poses_gt[i].t, poses_gt[i].pos[:,0], label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
            ax2.plot(poses_gt[i].t, poses_gt[i].pos[:,1], label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
            ax3.plot(poses_gt[i].t, poses_gt[i].pos[:,2], label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
        ax1.plot(poses_fused[i].t, poses_fused[i].pos[:,0], label=f"$D^2$VINS {_id}")
        ax2.plot(poses_fused[i].t, poses_fused[i].pos[:,1], label=f"$D^2$VINS {_id}")
        ax3.plot(poses_fused[i].t, poses_fused[i].pos[:,2], label=f"$D^2$VINS {_id}")
        if poses_pgo is not None:
            ax1.plot(poses_pgo[i].t, poses_pgo[i].pos[:,0], '.', label=f"$D^2$PGO Traj{i}")
            ax2.plot(poses_pgo[i].t, poses_pgo[i].pos[:,1], '.', label=f"$D^2$PGO Traj{i}")
            ax3.plot(poses_pgo[i].t, poses_pgo[i].pos[:,2], '.', label=f"$D^2$PGO Traj{i}")
        
                    
        ax1.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False) 
        ax1.set_ylabel("x")
        ax2.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False) 
        ax2.set_ylabel("y")
        ax3.set_ylabel("z")
        ax3.set_xlabel("t")
        ax3.legend()
        plt.savefig(output_path+f"est_by_t{i}_position.png")

        fig = plt.figure(f"Drone {i} fused Vs GT Vel", figsize=figsize)
        fig.suptitle(f"Drone {i} fused Vs GT 1D")
        ax1, ax2, ax3 = fig.subplots(3, 1)
        if poses_gt is not None:
            # Plot velocity
            ax1.plot(poses_gt[i].t, poses_gt[i].vel[:,0], label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
            ax2.plot(poses_gt[i].t, poses_gt[i].vel[:,1], label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
            ax3.plot(poses_gt[i].t, poses_gt[i].vel[:,2], label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
        ax1.plot(poses_fused[i].t, poses_fused[i].vel[:,0], label=f"$D^2$VINS {_id}")
        ax2.plot(poses_fused[i].t, poses_fused[i].vel[:,1], label=f"$D^2$VINS {_id}")
        ax3.plot(poses_fused[i].t, poses_fused[i].vel[:,2], label=f"$D^2$VINS {_id}")
        if poses_pgo is not None:
            ax1.plot(poses_pgo[i].t, poses_pgo[i].vel[:,0], '.', label=f"$D^2$PGO Traj{i}")
            ax2.plot(poses_pgo[i].t, poses_pgo[i].vel[:,1], '.', label=f"$D^2$PGO Traj{i}")
            ax3.plot(poses_pgo[i].t, poses_pgo[i].vel[:,2], '.', label=f"$D^2$PGO Traj{i}")
        ax1.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax1.set_ylabel("vx")
        ax2.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax2.set_ylabel("vy")
        ax3.set_ylabel("vz")
        ax3.set_xlabel("t")
        ax3.legend()
        plt.savefig(output_path+f"est_by_t{i}_velocity.png")

        fig = plt.figure(f"Drone {i} fused Vs GT Att", figsize=figsize)
        ax1, ax2, ax3 = fig.subplots(3, 1)

        if poses_gt is not None:
            ax1.plot(poses_gt[i].t, poses_gt[i].ypr[:,0]*57.3, label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
            ax2.plot(poses_gt[i].t, poses_gt[i].ypr[:,1]*57.3, label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
            ax3.plot(poses_gt[i].t, poses_gt[i].ypr[:,2]*57.3, label=f"Ground Truth ${i}$", marker='.', linestyle = 'None')
        if poses_pgo is not None:
            ax1.plot(poses_pgo[i].t, poses_pgo[i].ypr[:,0]*57.3, '.', label=f"$D^2$PGO {_id}")
            ax2.plot(poses_pgo[i].t, poses_pgo[i].ypr[:,1]*57.3, '.', label=f"$D^2$PGO {_id}")
            ax3.plot(poses_pgo[i].t, poses_pgo[i].ypr[:,2]*57.3, '.', label=f"$D^2$PGO {_id}")

        ax1.plot(poses_fused[i].t, poses_fused[i].ypr[:,0]*57.3, label=f"$D^2$VINS {_id}")
        ax2.plot(poses_fused[i].t, poses_fused[i].ypr[:,1]*57.3, label=f"$D^2$VINS {_id}")
        ax3.plot(poses_fused[i].t, poses_fused[i].ypr[:,2]*57.3, label=f"$D^2$VINS {_id}")

        ax1.set_ylabel("Yaw (deg)")
        ax1.set_xlabel("t")
        ax1.legend()
        ax2.set_ylabel("Pitch (deg)")
        ax2.set_xlabel("t")
        ax2.legend()
        ax3.set_xlabel("t")
        ax2.set_ylabel("Roll (deg)")
        ax3.legend()
        plt.savefig(output_path+f"est_by_t{i}_attitude.png")
        max_vel = np.max(np.linalg.norm(poses_fused[i].vel, axis=1))
        print(f"Trajetory {i} length {poses_fused[i].length():.1f} max vel: {max_vel:.1f}m/s")

def plot_relative_pose_err(main_id, target_ids, poses_fused, poses_gt, poses_vo=None,outlier_thres=100, 
        outlier_yaw_thres=10, dte=1000000, show=True, figsize=(6, 6), verbose=True, common_time_dt=0.2):
    if verbose:
        if poses_vo is not None:
            pass
        else:
            output_table = [["Relative", "EST RMSE: Pos (XYZ)", "POS", "Ang", "BIAS: Pos", "Ang"]]
    avg_rmse = 0
    avg_rmse_yaw = 0.0
    
    avg_rmse_vo = 0
    avg_rmse_vo_yaw = 0.0
    num = 0

    for target_id in target_ids:
        ts = find_common_times(poses_gt[main_id].t, poses_gt[target_id].t, dt=common_time_dt) #We need to find the common time period of these two
        ts = find_common_times(ts, poses_fused[main_id].t, dt=common_time_dt)
        ts = find_common_times(ts, poses_fused[target_id].t, dt=common_time_dt)
        ts = ts[ts<dte]
        if poses_vo is not None:
            posa_vo =  poses_vo[main_id].resample_pos(ts)
            yawa_vo = poses_vo[main_id].resample_ypr(ts)[:,0]
        posa_fused = poses_fused[main_id].resample_pos(ts)
        yawa_fused = poses_fused[main_id].resample_ypr(ts)[:,0]
        
        if poses_vo is not None:
            posb_vo =  poses_vo[target_id].resample_pos(ts)
            yawb_vo = poses_vo[target_id].resample_ypr(ts)[:, 0]
            dyaw_vo = wrap_pi(yawb_vo - yawa_vo)
            dp_vo = posb_vo - posa_vo
            for i in range(len(yawa_vo)):
                yaw = yawa_vo[i]
                dp_vo[i] = yaw_rotate_vec(-yaw, dp_vo[i])
            
        posb_fused = poses_fused[target_id].resample_pos(ts)
        yawb_fused = poses_fused[target_id].resample_ypr(ts)[:, 0]
        dp_fused = posb_fused - posa_fused
        dyaw_fused = wrap_pi(yawb_fused - yawa_fused)
        if poses_gt is not None:
            posa_gt =  poses_gt[main_id].resample_pos(ts)
            yawa_gt = poses_gt[main_id].resample_ypr(ts)[:, 0]
            posb_gt =  poses_gt[target_id].resample_pos(ts)
            yawb_gt = poses_gt[target_id].resample_ypr(ts)[:, 0]
            dp_gt = posb_gt - posa_gt
            dyaw_gt = wrap_pi(yawb_gt - yawa_gt)
        
        for i in range(len(yawa_fused)):
            yaw = yawa_fused[i]
            dp_fused[i] = yaw_rotate_vec(-yaw, dp_fused[i])

        if poses_gt is not None:
            for i in range(len(yawa_fused)):
                yaw = yawa_gt[i]
                dp_gt[i] = yaw_rotate_vec(-yaw, dp_gt[i])
        mask = np.linalg.norm(dp_gt - dp_fused, axis=1) < outlier_thres
        maskyaw = np.abs(wrap_pi(dyaw_gt - dyaw_fused)) < outlier_yaw_thres
        mask = np.logical_and(mask, maskyaw)
        if poses_gt is not None:
            rmse_yaw = RMSE(wrap_pi(dyaw_fused[mask] - dyaw_gt[mask]), 0)
            rmse_x = RMSE(dp_gt[mask,0] , dp_fused[mask,0])
            rmse_y = RMSE(dp_gt[mask,1] , dp_fused[mask,1])
            rmse_z = RMSE(dp_gt[mask,2] , dp_fused[mask,2])
            rmse_pos = ATE_POS(dp_gt[mask], dp_fused[mask])
            avg_rmse += rmse_pos
            avg_rmse_yaw += rmse_yaw
            num += 1

            if verbose:
                if poses_vo is not None:
                    pass
                else:
                    output_table.append([
                        f"{main_id}->{target_id}", f"{rmse_x:3.3f},{rmse_y:3.3f},{rmse_z:3.3f}", f"{rmse_pos:3.3f}", f"{rmse_yaw*180/pi:3.2f}°", 
                        f"{np.mean(dp_gt[mask,0] - dp_fused[mask,0]):3.3f},{np.mean(dp_gt[mask,1] - dp_fused[mask,1]):+3.3f},{np.mean(dp_gt[mask,2] - dp_fused[mask,2]):+3.3f}", 
                        f"{np.mean(dyaw_gt - dyaw_fused)*180/3.14:+3.2f}°"
                    ])
            if poses_vo is not None:
                rmse_yaw = RMSE(wrap_pi(yawb_vo - yawa_vo - yawb_gt + yawa_gt), 0)
                rmse_x = RMSE(dp_gt[mask,0] , dp_vo[mask,0])
                rmse_y = RMSE(dp_gt[mask,1] , dp_vo[mask,1])
                rmse_z = RMSE(dp_gt[mask,2] , dp_vo[mask,2])

                avg_rmse_vo += ATE_POS(dp_gt[mask], dp_vo[mask])
                avg_rmse_vo_yaw += rmse_yaw

        if show:
            fig = plt.figure(f"Relative Pose 2D {main_id}->{target_ids}", figsize=figsize)

            if poses_gt is not None:
                plt.plot(dp_gt[:, 0], dp_gt[:, 1], label=f"Relative Pose GT {main_id}->{target_id}")
            plt.plot(dp_fused[:, 0], dp_fused[:, 1], label=f"Relative Pose EST {main_id}->{target_id}")
            # plt.plot(dp_vo[:, 0], dp_vo[:, 1], label=f"Relative Pose VO {main_id}->{target_id}")
            plt.legend()
            if target_id == target_ids[0]:
                plt.grid()
            # Plot the histogram of relative pose of dp_fused in 3d
            plt.figure(f"Relative Pose Hist {main_id}->{target_id}", figsize=figsize)
            plt.hist2d(dp_fused[:, 0], dp_fused[:, 1], bins=10, range=[[-1, 1], [-1, 1]])
            plt.colorbar()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid()


            # fig = plt.figure("Relative Pose PolarErr", figsize=figsize)
            # fig.suptitle("Relative Pose PolarErr")
            # ax1, ax2 = fig.subplots(2, 1)

            # if poses_gt is not None:
            #     ax1.plot(ts[mask], wrap_pi(np.arctan2(dp_gt[:, 0], dp_gt[:, 1]) - np.arctan2(dp_fused[:, 0], dp_fused[:, 1]))[mask], label=f"Relative Pose Angular Err {main_id}->{target_id}")

            # if poses_gt is not None:
            #     ax2.plot(ts[mask], (norm(dp_gt, axis=1) - norm(dp_fused, axis=1))[mask], label=f"Relative Pose Length Err {main_id}->{target_id}")


            # ax1.legend()
            # ax1.grid()
            # ax2.legend()
            # ax2.grid()
            # plt.tight_layout()

            fig = plt.figure(f"Relative Pose {main_id}->{target_id}", figsize=figsize)
            fig.suptitle(f"Relative Pose {main_id}->{target_id}")
            ax1, ax2, ax3, ax4 = fig.subplots(4, 1)

            if poses_gt is not None:
                ax1.plot(ts, dp_gt[:,0], label="$X_{gt}^" + str(target_id) + "$")
                ax2.plot(ts, dp_gt[:,1], label="$Y_{gt}^" + str(target_id) + "$")
                ax3.plot(ts, dp_gt[:,2], label="$Z_{gt}^" + str(target_id) + "$")
                ax4.plot(ts, wrap_pi(dyaw_gt), label="$Yaw_{gt}^" + str(target_id) + "$")

            ax1.plot(ts, dp_fused[:,0], label="$X_{fused}^" + str(target_id) + "$")
            ax2.plot(ts, dp_fused[:,1], label="$Y_{fused}^" + str(target_id) + "$")
            ax3.plot(ts, dp_fused[:,2], label="$Z_{fused}^" + str(target_id) + "$")
            ax4.plot(ts, wrap_pi(dyaw_fused), label="$Yaw_{fused}^" + str(target_id) + "$")

            if poses_vo is not None:
                ax1.plot(ts, dp_vo[:,0], label="$X_{vo}^" + str(target_id) + "$")
                ax2.plot(ts, dp_vo[:,1], label="$Y_{vo}^" + str(target_id) + "$")
                ax3.plot(ts, dp_vo[:,2], label="$Z_{vo}^" + str(target_id) + "$")
                ax4.plot(ts, wrap_pi(dyaw_vo), label="$Yaw_{vo}^" + str(target_id) + "$")

            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax4.grid()
            plt.tight_layout()
                
            fig = plt.figure(f"Fused Relative Error {main_id}->{target_id}", figsize=figsize)
            fig.suptitle(f"Fused Relative Error {main_id}->{target_id}")
            ax1, ax2, ax3, ax4 = fig.subplots(4, 1)

            ax1.plot(ts[mask], dp_gt[mask,0] - dp_fused[mask,0], label="$E_{xfused}^" + str(target_id) + f"$ RMSE:{rmse_x:3.3f}")
            ax2.plot(ts[mask], dp_gt[mask,1] - dp_fused[mask,1], label="$E_{yfused}^" + str(target_id) + f"$ RMSE:{rmse_y:3.3f}")
            ax3.plot(ts[mask], dp_gt[mask,2] - dp_fused[mask,2], label="$E_{zfused}^" + str(target_id) + f"$ RMSE:{rmse_z:3.3f}")
            ax4.plot(ts[mask], wrap_pi(dyaw_gt[mask] - dyaw_fused[mask]), label="$E_{yawfused}^" + str(target_id) + f"$ RMSE:{rmse_z:3.3f}")


            fig = plt.figure(f"Fused Relative Distance vs Error {main_id}->{target_id}", figsize=figsize)
            fig.suptitle(f"Fused Relative Distance vs Error {main_id}->{target_id}")
            ax1, ax2= fig.subplots(2, 1)

            ax1.scatter(np.linalg.norm(dp_gt[mask,:3], axis=1), 
                        np.linalg.norm(dp_gt[mask,:3] - dp_fused[mask,:3], axis=1), marker='.', 
                        label="$E_{xfused}^" + str(target_id) + f"$ RMSE:{rmse_x:3.3f}")
            ax2.scatter(np.linalg.norm(dp_gt[mask,:3], axis=1), wrap_pi(dyaw_gt[mask] - dyaw_fused[mask]), marker='.',
                        label="$E_{yawfused}^" + str(target_id) + f"$ RMSE:{rmse_z:3.3f}")

            ax1.legend()
            ax2.legend()
            ax1.grid()
            ax2.grid()
            plt.tight_layout()
    output_table.append([
        "Avg:", f"", f"{avg_rmse/num:3.3f}", f"{avg_rmse_yaw/num*180/pi:3.2f}°", "", ""])
    if show:
        plt.show()
    if verbose:
        import tabulate
        return tabulate.tabulate(output_table, tablefmt='html')

def relative_pose_err(node_ids, poses_fused, poses_gt, outlier_thres=100, 
        outlier_yaw_thres=10, dte=1000000, common_time_dt=0.2, output_RE=False):
    output_table = [["Relative", "EST RMSE: Pos (XYZ)", "POS", "Ang", "BIAS: Pos", "Ang"]]
    avg_rmse = 0
    avg_rmse_yaw = 0.0
    num = 0
    for main_id in node_ids:
        for target_id in node_ids:
            if main_id > target_id:
                ts = find_common_times(poses_gt[main_id].t, poses_gt[target_id].t, dt=common_time_dt) #We need to find the common time period of these two
                ts = find_common_times(ts, poses_fused[main_id].t, dt=common_time_dt)
                ts = find_common_times(ts, poses_fused[target_id].t, dt=common_time_dt)
                ts = ts[ts<dte]
                posa_fused = poses_fused[main_id].resample_pos(ts)
                yawa_fused = poses_fused[main_id].resample_ypr(ts)[:,0]
                
                posb_fused = poses_fused[target_id].resample_pos(ts)
                yawb_fused = poses_fused[target_id].resample_ypr(ts)[:, 0]
                dp_fused = posb_fused - posa_fused
                dyaw_fused = wrap_pi(yawb_fused - yawa_fused)
                if poses_gt is not None:
                    posa_gt =  poses_gt[main_id].resample_pos(ts)
                    yawa_gt = poses_gt[main_id].resample_ypr(ts)[:, 0]
                    posb_gt =  poses_gt[target_id].resample_pos(ts)
                    yawb_gt = poses_gt[target_id].resample_ypr(ts)[:, 0]
                    dp_gt = posb_gt - posa_gt
                    dyaw_gt = wrap_pi(yawb_gt - yawa_gt)
                
                for i in range(len(yawa_fused)):
                    yaw = yawa_fused[i]
                    dp_fused[i] = yaw_rotate_vec(-yaw, dp_fused[i])

                if poses_gt is not None:
                    for i in range(len(yawa_fused)):
                        yaw = yawa_gt[i]
                        dp_gt[i] = yaw_rotate_vec(-yaw, dp_gt[i])
                mask = np.linalg.norm(dp_gt - dp_fused, axis=1) < outlier_thres
                maskyaw = np.abs(wrap_pi(dyaw_gt - dyaw_fused)) < outlier_yaw_thres
                mask = np.logical_and(mask, maskyaw)
                if poses_gt is not None:
                    rmse_yaw = RMSE(wrap_pi(dyaw_fused[mask] - dyaw_gt[mask]), 0)
                    rmse_x = RMSE(dp_gt[mask,0] , dp_fused[mask,0])
                    rmse_y = RMSE(dp_gt[mask,1] , dp_fused[mask,1])
                    rmse_z = RMSE(dp_gt[mask,2] , dp_fused[mask,2])
                    rmse_pos = ATE_POS(dp_gt[mask], dp_fused[mask])
                    avg_rmse += rmse_pos
                    avg_rmse_yaw += rmse_yaw
                    num += 1

                    output_table.append([
                        f"{main_id}->{target_id}", f"{rmse_x:3.3f},{rmse_y:3.3f},{rmse_z:3.3f}", f"{rmse_pos:3.3f}", f"{rmse_yaw*180/pi:3.2f}°", 
                        f"{np.mean(dp_gt[mask,0] - dp_fused[mask,0]):3.3f},{np.mean(dp_gt[mask,1] - dp_fused[mask,1]):+3.3f},{np.mean(dp_gt[mask,2] - dp_fused[mask,2]):+3.3f}", 
                        f"{np.mean(dyaw_gt - dyaw_fused)*180/3.14:+3.2f}°"
                    ])
    if output_RE:
        return avg_rmse/num, avg_rmse_yaw/num
    output_table.append([
        "Avg:", f"", f"{avg_rmse/num:3.3f}", f"{avg_rmse_yaw/num*180/pi:3.2f}°", "", ""])
    import tabulate
    return tabulate.tabulate(output_table, tablefmt='html')
        
def plot_fused_err(nodes, poses_fused, poses_gt, poses_vo=None, poses_pgo=None,main_id=1,dte=100000,show=True, 
    outlier_thres=100, outlier_thres_yaw=100, verbose=True, output_ATE=False):
    #Plot Fused Vs GT absolute error
    ate_vo_sum = 0
    rmse_vo_ang_sum = 0

    ate_fused_sum = 0
    rmse_fused_ang_sum = 0
    ate_pgo_sum = 0
    rmse_pgo_ang_sum = 0
    output_table = [["Drone", "Traj. Len.", "ATE Pos", "ATE Att", "Cov/m: x", "y", "z", "Cov Att/m", "PGO:ATE Pos ", "ATE Att"]]

    ate_pos_sum = 0
    ate_ang_sum = 0
    length_sum = 0
    num = 0
    for i in nodes:
        t_ = find_common_times(poses_gt[i].t, poses_fused[i].t)
        t_ = t_[t_<dte]
        pos_gt =  poses_gt[i].resample_pos(t_)
        pos_fused = poses_fused[i].resample_pos(t_)
        ypr_fused = poses_fused[i].resample_ypr(t_)
        ypr_gt = poses_gt[i].resample_ypr(t_)
        if poses_vo is not None:
            pos_vo = poses_vo[i].resample_pos(t_)
            ypr_vo = poses_vo[i].resample_ypr(t_)
        
        mask_fused = np.linalg.norm(pos_fused - pos_gt, axis=1) < outlier_thres
        mask_fused = np.logical_and(mask_fused, np.linalg.norm(ypr_gt - ypr_fused, axis=1) < outlier_thres_yaw)
        pos_gt =  pos_gt[mask_fused]
        ypr_gt = ypr_gt[mask_fused]
        pos_fused = pos_fused[mask_fused]
        ypr_fused = ypr_fused[mask_fused]
        t_ = t_[mask_fused]

        fused_cov_per_meter, fused_yaw_cov_per_meter = odometry_covariance_per_meter(pos_fused, ypr_fused[:,0], pos_gt, ypr_gt[:,0])
        rmse_x = RMSE(pos_fused[:,0] , pos_gt[:,0])
        rmse_y = RMSE(pos_fused[:,1] , pos_gt[:,1])
        rmse_z = RMSE(pos_fused[:,2] , pos_gt[:,2])

        fused_cov_x = fused_cov_per_meter[0][0]
        fused_cov_y = fused_cov_per_meter[1][1]
        fused_cov_z = fused_cov_per_meter[2][2]

        if np.isnan(pos_fused).any():
            print("pos_fused has nan")
        if np.isnan(pos_gt).any():
            print("pos_gt has nan")
        
        ate_fused = ATE_POS(pos_fused, pos_gt)
        rmse_yaw_fused = RMSE(wrap_pi(ypr_gt[:,0]-ypr_fused[:,0]), 0)
        rmse_pitch_fused = RMSE(wrap_pi(ypr_gt[:,1]-ypr_fused[:,1]), 0)
        rmse_roll_fused = RMSE(wrap_pi(ypr_gt[:,2]-ypr_fused[:,2]), 0)
        rmse_angular_fused = RMSE(angular_error_ypr_array(ypr_gt, ypr_fused), 0)

        ate_fused_sum += ate_fused
        rmse_fused_ang_sum += rmse_angular_fused
        
        if poses_pgo is not None:
            t_pgo = find_common_times(poses_gt[i].t, poses_pgo[i].t)
            pos_path_gt =  poses_gt[i].resample_pos(t_pgo)
            pos_path = poses_pgo[i].resample_pos(t_pgo)
            ypr_path_gt =  poses_gt[i].resample_ypr(t_pgo)
            ypr_path = poses_pgo[i].resample_ypr(t_pgo)
            mask_path = np.linalg.norm(pos_path_gt - pos_path, axis=1) < outlier_thres
            t_path = t_pgo[mask_path]
            pos_path_gt, pos_path, ypr_path_gt, ypr_path = pos_path_gt[mask_path], pos_path[mask_path], ypr_path_gt[mask_path], ypr_path[mask_path]

            ate_path = ATE_POS(pos_path, pos_path_gt)
            rmse_angular_path = RMSE(angular_error_ypr_array(ypr_path_gt, ypr_path), 0)
            ate_pgo_sum += ate_path
            rmse_pgo_ang_sum += rmse_angular_path
        else:
            ate_path = nan
            rmse_angular_path = nan

        if poses_vo is not None:
            pos_gt_vo=  poses_gt[i].resample_pos(t_)
            ypr_gt_vo =  poses_gt[i].resample_ypr(t_)
            mask_vo = np.linalg.norm(pos_gt_vo - pos_vo, axis=1) < outlier_thres
            pos_vo = pos_vo[mask_vo]
            pos_gt_vo = pos_gt_vo[mask_vo]
            ypr_vo = ypr_vo[mask_vo]
            ypr_gt_vo = ypr_gt_vo[mask_vo]
            t_vo = t_[mask_vo]
            rmse_vo_x = RMSE(pos_vo[:,0] , pos_gt_vo[:,0])
            rmse_vo_y = RMSE(pos_vo[:,1] , pos_gt_vo[:,1])
            rmse_vo_z = RMSE(pos_vo[:,2] , pos_gt_vo[:,2])
            ate_vo = ATE_POS(pos_vo, pos_gt_vo)
            # vo_cov_per_meter, vo_yaw_cov_per_meter = odometry_covariance_per_meter(pos_vo, ypr_vo[:,0], pos_gt_vo, ypr_gt_vo[:,0], show=False)
            # rmse_yaw_vo = RMSE(wrap_pi(ypr_vo[:,0]-ypr_gt_vo[:,0]), 0)
            # rmse_pitch_vo = RMSE(wrap_pi(ypr_vo[:,1]-ypr_gt_vo[:,1]), 0)
            # rmse_roll_vo = RMSE(wrap_pi(ypr_vo[:,2]-ypr_gt_vo[:,2]), 0)
            rmse_angular_vo = RMSE(angular_error_ypr_array(ypr_vo, ypr_gt_vo), 0)

            ate_vo_sum += ate_vo
            rmse_vo_ang_sum += rmse_angular_vo

            if verbose:
                pass
        traj_len = poses_fused[i].length(dte)
        ate_pos_sum += ate_fused
        ate_ang_sum += rmse_angular_fused
        length_sum += traj_len
        num += 1
        output_table.append([
            f"{i}",f"{traj_len:.1f}m", f"{ate_fused:3.3f}", f"{rmse_angular_fused*180/pi:3.3f}", 
            f"{fused_cov_x:.1e}",f"{fused_cov_y:.1e}",f"{fused_cov_z:.1e}",
            f"{fused_yaw_cov_per_meter*180/pi:.2e}",
            f"{ate_path:3.3f}",f"{rmse_angular_path*180/pi:3.3f}°"
        ])

        if show:
            fig = plt.figure(f"Fused Absolute Error Pos {i}")
            fig.suptitle(f"Fused Absolute Error Pos {i}")
            ax1, ax2, ax3 = fig.subplots(3, 1)
            label = f"$errx_{i}$ RMSE{i}:{rmse_x:3.3f}"
            ax1.plot(t_, pos_gt[:,0]  - pos_fused[:,0], label=label)

            label = f"$erry_{i}$ RMSE{i}:{rmse_y:3.3f}"
            ax2.plot(t_, pos_gt[:,1]  - pos_fused[:,1], label=label)

            label = f"$erry_{i}$ RMSE{i}:{rmse_z:3.3f}"
            ax3.plot(t_,  pos_gt[:,2]  - pos_fused[:,2], label=label)

            if poses_vo is not None:
                label = f"$VO errx_{i}$ RMSE{i}:{rmse_vo_x:3.3f}"
                ax1.plot(t_vo, pos_gt_vo[:,0]  - pos_vo[:,0], label=label)

                label = f"$VO erry_{i}$ RMSE{i}:{rmse_vo_y:3.3f}"
                ax2.plot(t_vo, pos_gt_vo[:,1]  - pos_vo[:,1], label=label)
                
                label = f"$VO errz_{i}$ RMSE{i}:{rmse_vo_z:3.3f}"
                ax3.plot(t_vo, pos_gt_vo[:,2]  - pos_vo[:,2], label=label)

            if poses_pgo is not None:
                label = f"$PGO errx_{i}$"
                ax1.plot(t_path, pos_path_gt[:,0]  - pos_path[:,0], label=label)

                label = f"$PGO erry_{i}$"
                ax2.plot(t_path, pos_path_gt[:,1]  - pos_path[:,1], label=label)
                
                label = f"$PGO errz_{i}$"
                ax3.plot(t_path, pos_path_gt[:,2]  - pos_path[:,2], label=label)

            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax1.grid()
            ax2.grid()
            ax3.grid()

            fig = plt.figure(f"Fused Absolute Error Att {i}")
            fig.suptitle(f"Fused Absolute Error Att {i}")
            ax1, ax2, ax3 = fig.subplots(3, 1)
            label = f"$VO yaw_{i}$ RMSE{i}:{rmse_z:3.3f}"
            
            
            label = f"$yaw_{i}$ RMSE{i}:{rmse_z:3.3f}"
            ax1.plot(t_, wrap_pi(ypr_gt[:,0]-ypr_fused[:,0]), label=label)

            label = f"$pitch_{i}$ RMSE{i}:{rmse_z:3.3f}"
            ax2.plot(t_, wrap_pi(ypr_gt[:,1]-ypr_fused[:,1]), label=label)

            label = f"$roll_{i}$ RMSE{i}:{rmse_z:3.3f}"
            ax3.plot(t_, wrap_pi(ypr_gt[:,2]-ypr_fused[:,2]), label=label)

            if poses_vo is not None:
                ax1.plot(t_vo, wrap_pi(ypr_gt_vo[:,0]-ypr_vo[:,0]), label=label)
                label = f"$VO pitch_{i}$ RMSE{i}:{rmse_z:3.3f}"
                ax2.plot(t_vo, wrap_pi(ypr_gt_vo[:,1]-ypr_vo[:,1]), label=label)
                label = f"$VO roll_{i}$ RMSE{i}:{rmse_z:3.3f}"
                ax3.plot(t_vo, wrap_pi(ypr_gt_vo[:,2]-ypr_vo[:,2]), label=label)
            
            if poses_pgo is not None:
                label = f"$Path yaw_{i}$"
                ax1.plot(t_path, wrap_pi(ypr_path_gt[:,0]  - ypr_path[:,0]), ".", label=label)
                label = f"$Path pitch_{i}$"
                ax2.plot(t_path, wrap_pi(ypr_path_gt[:,1]  - ypr_path[:,1]), ".", label=label)
                label = f"$Path roll_{i}$"
                ax3.plot(t_path, wrap_pi(ypr_path_gt[:,2]  - ypr_path[:,2]), ".", label=label)
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax1.grid()
            ax2.grid()
            ax3.grid()

    output_table.append([
        f"Avg.",f"{length_sum/num:.1f}", f"{ate_fused_sum/num:3.3f}", f"{ate_ang_sum/num*180/pi:3.2f}", "","","",
            "", f"{ate_pgo_sum/num:.3f}",f"{rmse_pgo_ang_sum/num*180/pi:3.2f}"])
    if poses_pgo is None:
        #Remove the last two columns of output table
        print("no pgo")
        output_table = [row[:-2] for row in output_table]
    if output_ATE:
        return ate_fused_sum/num, ate_ang_sum/num
    if verbose:
        import tabulate
        return tabulate.tabulate(output_table, tablefmt='html')

def plot_detections_error(poses, poses_vo, detections, main_id, enable_dpose, inlier_file=""):
    _dets_data = []
    dpos_dets = []
    dpos_gts = []
    dpos_gt_norms= []
    dpos_det_norms= []
    dpos_errs = []
    inv_dep_errs = []
    inv_deps = []
    dpos_errs_norm = []
    posa_gts = []
    ts_a = []
    dyaws = []
    yawa_gts = []
    self_pos_a = []
    self_pos_b = []
    inv_deps_gt = []
    det_ids = []
    print("Total detection", len(detections))
    CG = np.array([-0.06, 0, 0])
    good_det_id = set()
    if inlier_file != "":
        with open(inlier_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                good_det_id.add(int(line))


    for det in detections:
        if det["id_a"] != main_id:
            continue
        yawa_gt = poses[det["id_a"]].resample_ypr(det["ts"])[0]
        yawb_gt = poses[det["id_b"]].resample_ypr(det["ts"])[0]

        posa_gt = poses[det["id_a"]].resample_pos(det["ts"])
        posb_gt = poses[det["id_b"]].resample_pos(det["ts"]) # + yaw_rotate_vec(yawb_gt, np.array([-0.04, 0, 0.02]))

        if enable_dpose:
            # dpose_self_a = yaw_rotate_vec(yawa_gt, yaw_rotate_vec(-yawa_vo, posa_vo - det["pos_a"]))
            posa_gt = posa_gt + yaw_rotate_vec(yawa_gt, det["extrinsic"])
            posb_gt = posb_gt + yaw_rotate_vec(yawb_gt, CG)

        dpos_gt = yaw_rotate_vec(-yawa_gt, posb_gt - posa_gt)
        inv_dep_gt = 1/norm(dpos_gt)
        dpos_gt = dpos_gt * inv_dep_gt
        
        dpos_det = np.array(det["dpos"])
        inv_dep_det = det["inv_dep"]
        _dets_data.append({
            "dpos_det": dpos_det,
            "dpos_gt": dpos_gt,
            "dpos_err": dpos_gt - dpos_det,
            "inv_dep_err": inv_dep_gt - inv_dep_det
            })
        #TMP
        inv_deps.append(det["inv_dep"])
        self_pos_a.append(det["pos_a"])
        self_pos_b.append(det["pos_b"])

        inv_dep_errs.append(inv_dep_gt - inv_dep_det)
        inv_deps_gt.append(inv_dep_gt)
        dpos_dets.append(dpos_det)
        dpos_gts.append(dpos_gt)
        dpos_errs.append(dpos_gt - dpos_det)    
        dpos_gt_norms.append(norm(dpos_gt))
        dpos_det_norms.append(norm(dpos_det))
        dpos_errs_norm.append(norm(dpos_gt - dpos_det))
        posa_gts.append(posa_gt)
        ts_a.append(det["ts"])
        yawa_gts.append(yawa_gt)
        det_ids.append(det["id"])

        # pa = det['pos_a']
        # pb = det["pos_b"]
        # dp = det["dpos"]
        # print(f"Det {det['id_a']} -> {det['id_b']}")
        # print(f"SELF POSE A {pa} B {pb} p {dp}")
        # print(f"POSE A {posa_gt} B {posb_gt} PB-PA {posb_gt-posa_gt}")
        # print(f"det dp {dpos_det} est dp{dpos_gt} yawa {yawa_gt*57.3}deg")
        
    posa_gts = np.array(posa_gts)
    dpos_errs = np.array(dpos_errs)
    dpos_gts = np.array(dpos_gts)
    dpos_dets = np.array(dpos_dets)
    self_pos_a = np.array(self_pos_a)
    self_pos_b = np.array(self_pos_b)
    ts_a = np.array(ts_a)
    fig = plt.figure()

    plt.subplot(311)
    plt.plot(ts_a, dpos_errs_norm, '.', label="Err NORM")
    plt.plot(ts_a, np.abs(dpos_errs[:,0]), '+', label="ERR X")
    plt.plot(ts_a, np.abs(dpos_errs[:,1]), '+', label="ERR Y")
    plt.plot(ts_a, dpos_errs[:,2], '+', label="ERR Z")
    # plt.ylim((-0.2, 0.2))
    plt.title("Error Pos Detection vs Vicon")
    plt.grid(which="both")
    plt.legend()

    plt.subplot(312)
    plt.plot(ts_a, dpos_gts[:,0], '.', label="GT X")
    plt.plot(ts_a, dpos_gts[:,1], '.', label="GT Y")
    plt.plot(ts_a, dpos_gts[:,2], '.', label="GT Z")
    
    plt.plot(ts_a, dpos_dets[:,0], '+', label="Detection X")
    plt.plot(ts_a, dpos_dets[:,1], '+', label="Detection Y")
    plt.plot(ts_a, dpos_dets[:,2], '+', label="Detection Z")
    plt.legend()
    plt.grid()
    plt.subplot(313)
    plt.plot(ts_a, inv_dep_errs, '.', label="INV DEP ERR")

    plt.grid(which="both")
    plt.legend()

    plt.figure("Direction Err hist")
    plt.subplot(131)
    plt.hist(dpos_errs[:,0], 50, (-0.1, 0.1), density=True, facecolor='g', alpha=0.75)
    xmin, xmax = plt.xlim()
    
    mu, std = stats.norm.fit(dpos_errs[:,0])
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "X mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    plt.subplot(132)
    plt.hist(dpos_errs[:,1], 50, (-0.1, 0.1), density=True, facecolor='g', alpha=0.75)
    xmin, xmax = plt.xlim()

    mu, std = stats.norm.fit(dpos_errs[:,1])
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Y mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    plt.subplot(133)
    plt.hist(dpos_errs[:,2], 50, (-0.1, 0.1), density=True, facecolor='g', alpha=0.75)
    xmin, xmax = plt.xlim()
    
    mu, std = stats.norm.fit(dpos_errs[:,2])
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Z mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    filter_dpos = np.array(dpos_errs_norm) < 0.2
    print(f"Mean {np.mean(dpos_errs[filter_dpos], axis=0)}")

    print("Pos cov", np.cov(dpos_errs[:,0]), np.cov(dpos_errs[:,1]), np.cov(dpos_errs[:,2]) )


    inv_dep_err = np.array(inv_deps) - np.array(inv_deps_gt)
    mask = np.fabs(inv_dep_err) < 0.3
    plt.figure("INV DEPS")
    plt.title("INV DEPS")
    plt.plot(ts_a, np.array(inv_deps), "+", label="INV DEP DET")
    plt.plot(ts_a, np.array(inv_deps_gt), "x", label="INV DEP GT")

    if len(good_det_id) > 0:
        for i in range(len(ts_a)):
            if det_ids[i] not in good_det_id:
                plt.text(ts_a[i], inv_deps[i], "x", color="red")

    plt.legend()
    plt.grid()

    plt.figure("INV DEPS ERR Inliers")
    plt.title("INV DEPS ERR Inliers")
    plt.plot(ts_a[mask], inv_dep_err[mask], "+", label="INV DEP DET")

    plt.legend()
    plt.grid()


    plt.figure("INV DEPS ERR HIST of inliners")
    plt.hist(np.array(inv_deps)[mask] - np.array(inv_deps_gt)[mask], 50, (-0.3, 0.3), density=True, facecolor='g', alpha=0.75)
    mu, std = stats.norm.fit(np.array(inv_deps)[mask] - np.array(inv_deps_gt)[mask])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "INV DEPS ERR  Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    print("INV DEPS ERR Variance", np.mean((np.array(inv_deps) - np.array(inv_deps_gt))**2))

    plt.title(title)
    plt.legend()
    plt.grid()


def plot_loops_error(poses, loops, good_loop_id=None, outlier_show_thres=0.5, show_outlier=True):
    _loops_data = []
    dpos_loops = []
    dpos_gts = []
    dpos_gt_norms= []
    dpos_loop_norms= []
    dpos_errs = []
    dpos_errs_norm = []
    posa_gts = []
    distances = []
    ts_a = []
    dyaws = []
    dyaw_gts = []
    yawa_gts = []
    yawb_gts = []
    dyaw_errs = []
    pnp_inlier_nums = []
    idas = []
    idbs = []
    dts = []
    count_inter_loop = 0

    loops_error = {}
    loop_ids = []
    for loop in loops:
        # print(loop["id_a"], "->", loop["id_b"])
        if loop["id_a"] != loop["id_b"]:
            count_inter_loop += 1
        if loop["id_a"] not in poses or loop["id_b"] not in poses:
            continue
        posa_gt = poses[loop["id_a"]].resample_pos(loop["ts_a"])
        posb_gt = poses[loop["id_b"]].resample_pos(loop["ts_b"])
        yawa_gt = poses[loop["id_a"]].resample_ypr(loop["ts_a"])[0]
        yawb_gt = poses[loop["id_b"]].resample_ypr(loop["ts_b"])[0]
        dpos_gt = yaw_rotate_vec(-yawa_gt, posb_gt - posa_gt)
        dpos_loop = np.array(loop["dpos"])
        if np.linalg.norm(dpos_gt - dpos_loop) > outlier_show_thres and good_loop_id is not None and loop["id"] not in good_loop_id:
            continue
        _loops_data.append({
            "dpos_loop": dpos_loop,
            "dpos_gt": dpos_gt,
            "dpos_err": dpos_gt - dpos_loop})
        dpos_loops.append(dpos_loop)
        dpos_gts.append(dpos_gt)
        dpos_errs.append(dpos_gt - dpos_loop)    
        dpos_gt_norms.append(norm(dpos_gt))
        dpos_loop_norms.append(norm(dpos_loop))
        dpos_errs_norm.append(norm(dpos_gt - dpos_loop))
        
        posa_gts.append(posa_gt)
        dyaws.append(wrap_pi(loop["dyaw"]))
        dyaw_gts.append(wrap_pi(yawb_gt-yawa_gt))
        if loop["ts_a"] > loop["ts_b"]:
            ts_a.append(loop["ts_a"])
        else:
            ts_a.append(loop["ts_b"])
        yawa_gts.append(wrap_pi(yawa_gt))
        yawb_gts.append(wrap_pi(yawb_gt))
        dyaw_errs.append(wrap_pi(yawb_gt-yawa_gt-loop["dyaw"]))
        pnp_inlier_nums.append(loop["pnp_inlier_num"])
        idas.append(loop["id_a"])
        idbs.append(loop["id_b"])
        loop_ids.append(loop["id"])
        dts.append(fabs(loop["ts_b"]-loop["ts_a"]))

        if loop["id"] in loops_error:
            print("Duplicate loop", loop["id"])
        else:
            loops_error[loop["id"]] = {
                "ida": loop["id_a"],
                "idb": loop["id_b"],
                "gt_pos": dpos_gt, 
                "gt_pos_a": posa_gt,
                "gt_pos_b": posb_gt,
                "est_pos": dpos_loop, 
                "err_pos": norm(dpos_gt - dpos_loop), 
                "err_yaw": wrap_pi(yawb_gt-yawa_gt-loop["dyaw"]), 
                "dt": fabs(loop["ts_b"]-loop["ts_a"])
            }
        # if np.linalg.norm(dpos_gt - dpos_loop) > 1.0:
            # print(loop["id"], loops_error[loop["id"]])
    
    outlier_num = (np.array(dpos_errs_norm)>0.5).sum()
    total_loops = len(dpos_errs_norm)
    print(f"Outlier rate {outlier_num/total_loops*100:3.2f}% total loops {total_loops} inter_loops {count_inter_loop} outlier_num {outlier_num}")
    posa_gts = np.array(posa_gts)
    dpos_errs = np.array(dpos_errs)
    dyaw_errs = np.array(dyaw_errs)
    distances = np.array(distances)
    dpos_loops = np.array(dpos_loops)
    dpos_gts = np.array(dpos_gts)
    dyaws = np.array(dyaws)
    dyaw_gts = np.array(dyaw_gts)
    dpos_loop_norms = np.array(dpos_loop_norms)
    dpos_errs_norm = np.array(dpos_errs_norm)
    dts = np.array(dts)

    fig = plt.figure("Loop Error")
    plt.subplot(211)
    plt.tight_layout()
    plt.plot(ts_a, dpos_errs_norm, '.', label="Loop Error")
    plt.plot(ts_a, dpos_errs[:,0], '.', label="Loop Error X")
    plt.plot(ts_a, dpos_errs[:,1], '.', label="Loop Error Y")
    plt.plot(ts_a, dpos_errs[:,2], '.', label="Loop Error Z")
    for i in range(len(loop_ids)):
        if (good_loop_id is not None and loop_ids[i] not in good_loop_id) or dpos_errs_norm[i] > outlier_show_thres:
            plt.text(ts_a[i], dpos_errs_norm[i], f"x{short_loop_id(loop_ids[i])}", fontsize=12, color="red")

    plt.title(f"Error Pos Loop vs Vicon. ErrNorm max {np.max(dpos_errs_norm):.2f}m")
    plt.ylim(-np.min(dpos_errs_norm)*1.2, np.max(dpos_errs_norm)*1.2)
    plt.grid(which="both")
    plt.legend()

    plt.subplot(212)
    plt.plot(ts_a, dyaws*57.3, '.', label="DYaw Gt")
    plt.plot(ts_a, dyaw_gts*57.3, '+', label="DYaw Loop")
    plt.plot(ts_a, np.abs(dyaw_errs)*57.3, "x", label="DYaw Error")
    
    plt.title("Loop Yaw (deg)")
    plt.grid(which="both")
    plt.legend()

    # plt.subplot(313)
    # plt.plot(ts_a, pnp_inlier_nums, "x", label="pnp_inlier_nums")
    # plt.grid()

    fig = plt.figure("Loop Comp")

    plt.subplot(311)
    plt.plot(ts_a, dpos_loops[:,0], '+', label="RelPose Est")
    plt.plot(ts_a, dpos_gts[:,0], '.', label="RelPose GT")
    plt.grid(which="both")
    plt.ylabel("X")
    plt.legend()

    plt.subplot(312)
    plt.plot(ts_a, dpos_loops[:,1], '+', label="RelPose Est")
    plt.plot(ts_a, dpos_gts[:,1], '.', label="RelPose GT")
    plt.grid(which="both")
    plt.legend()
    plt.ylabel("Y")

    plt.subplot(313)
    plt.plot(ts_a, dpos_loops[:,2], '+', label="RelPose Est")
    plt.plot(ts_a, dpos_gts[:,2], '.', label="RelPose GT")
    plt.ylabel("Z")

    plt.grid(which="both")
    plt.legend()

    # plt.figure("InliersVSErr")
    # plt.title("InliersVSErr")
    # plt.plot(pnp_inlier_nums, dpos_errs_norm, "x", label="")
    # plt.grid(which="both")
    # for i in range(len(pnp_inlier_nums)):
    #     if dpos_errs_norm[i]>0.2:
    #         plt.text(pnp_inlier_nums[i], dpos_errs_norm[i], f"{short_loop_id(loop_ids[i])}|{idas[i]}->{idbs[i]}", fontsize=12)

    plt.figure("Distance vs PosErr")
    plt.title("Distance vs PosErr")
    plt.subplot(221)
    plt.plot(dpos_loop_norms, dpos_errs_norm, ".", label="")
    plt.grid(which="both")

    mask = []
    if good_loop_id is not None:
        for i in range(len(loop_ids)):
            mask.append(loop_ids[i] in good_loop_id)

    for i in range(len(pnp_inlier_nums)):
        if good_loop_id is not None and loop_ids[i] not in good_loop_id:
            plt.text(dpos_loop_norms[i], dpos_errs_norm[i], f"x{short_loop_id(loop_ids[i])}", fontsize=12, color="red")
        elif dpos_errs_norm[i]>outlier_show_thres:
            plt.text(dpos_loop_norms[i], dpos_errs_norm[i], f"{short_loop_id(loop_ids[i])}", fontsize=12)

    plt.subplot(222)
    plt.plot(dpos_loop_norms[mask], dpos_errs_norm[mask], ".", label="")
    plt.grid(which="both")


    plt.subplot(223)
    plt.plot(dpos_loop_norms, dyaw_errs*57.3, ".", label="")
    plt.grid(which="both")
    for i in range(len(pnp_inlier_nums)):
        if good_loop_id is not None and loop_ids[i] not in good_loop_id:
            plt.text(dpos_loop_norms[i], dyaw_errs[i]*57.3, f"x{short_loop_id(loop_ids[i])}", fontsize=12, color="red")
        elif dpos_errs_norm[i]>outlier_show_thres:
            plt.text(dpos_loop_norms[i], dyaw_errs[i]*57.3, f"{short_loop_id(loop_ids[i])}", fontsize=12)

    plt.subplot(224)
    plt.plot(dpos_loop_norms[mask], dyaw_errs[mask]*57.3, ".", label="")
    plt.grid(which="both")

    plt.figure("Dt vs YawErr (deg)")
    plt.plot(ts_a, dpos_errs_norm, ".", label="")
    for i in range(len(dts)):
        if good_loop_id is not None and loop_ids[i] not in good_loop_id:
            plt.text(ts_a[i], dpos_errs_norm[i], f"x{short_loop_id(loop_ids[i])}", fontsize=12, color="red")
    plt.grid()

    if good_loop_id is not None:
        dpos_errs=dpos_errs[mask]
        dyaw_errs = dyaw_errs[mask]

    plt.figure("Loop Hist")
    plt.subplot(141)
    plt.hist(dpos_errs[:,0], 50, density=True, facecolor='g', alpha=0.75)

    mu, std = stats.norm.fit(dpos_errs[:,0])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "mu = %.2e,  std = %.2e\ncov(mu=0) = %.2e" % (mu, std, RMSE(dpos_errs[:,0], 0)**2)
    plt.title(title)

    plt.subplot(142)
    plt.hist(dpos_errs[:,1], 50, density=True, facecolor='g', alpha=0.75)
    mu, std = stats.norm.fit(dpos_errs[:,1])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "mu = %.2e,  std = %.2e\ncov(mu=0) = %.2e" % (mu, std, RMSE(dpos_errs[:,1], 0)**2)
    plt.title(title)

    plt.subplot(143)
    plt.hist(dpos_errs[:,2], 50, density=True, facecolor='g', alpha=0.75)
    mu, std = stats.norm.fit(dpos_errs[:,2])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "mu = %.2e,  std = %.2e\ncov(mu=0) = %.2e" % (mu, std, RMSE(dpos_errs[:,2], 0)**2)
    plt.title(title)

    plt.subplot(144)
    plt.hist(dyaw_errs, 50, density=True, facecolor='g', alpha=0.75)
    mu, std = stats.norm.fit(dpos_errs[:,2])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "mu = %.2e,  std = %.2e\ncov(mu=0) = %.2e" % (mu, std, RMSE(dpos_errs[:,2], 0)**2)
    plt.title(title)

    print(f"Pos cov {np.cov(dpos_errs[:,0]):.1e}, {np.cov(dpos_errs[:,1]):.1e}, {np.cov(dpos_errs[:,2]):.1e}")
    print(f"Yaw cov {np.cov(dyaw_errs):.1e}")

    print(f"Pos std {np.sqrt(np.cov(dpos_errs[:,0])):.1e}, {np.sqrt(np.cov(dpos_errs[:,1])):.1e}, {np.sqrt(np.cov(dpos_errs[:,2])):.1e}")
    print(f"Yaw std {np.sqrt(np.cov(dyaw_errs)):.1e}")

    return loops_error