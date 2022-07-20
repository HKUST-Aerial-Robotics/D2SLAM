#!/usr/bin/env python
from glob import glob
from pose_graph_partitioning.pose_graph import *
from pose_graph_partitioning.tsp_dataset_generation import *
from pose_graph_partitioning.pose_graph_partitioning import *
from simulate_bdslam_realworld import *
import pickle
import os
import rosbag
import simulate_bdslam_realworld as bdslam
from swarmcomm_msgs.msg import *

VO_POS_COV_PER_METER = 0.002
VO_ATT_COV_PER_METER = 0.00005

def read_network_status(self_id, bag, t0, start_id):
    network_status = []
    for topic, msg, t in bag.read_messages(topics=['/swarm_drones/swarm_network_status']):
        node_ids = []
        for i in range(len(msg.node_ids)):
            node_ids.append(msg.node_ids[i] - start_id)
            msg.network_status[i].drone_id = msg.network_status[i].drone_id - start_id
        msg.node_ids = node_ids
        network_status.append((t.to_sec() - t0, self_id-start_id, msg))
    return network_status

def create_connected_network_statues(drone_ids):
    network_status = []
    #Generate connected at first to avoid some bugs.
    for _id in drone_ids:
        nstats = swarm_network_status()
        for _idj in drone_ids:
            if _id != _idj:
                dstatus = drone_network_status()
                dstatus.active = True
                dstatus.bandwidth = -1
                dstatus.drone_id = _idj
                nstats.node_ids.append(_idj)
                nstats.network_status.append(dstatus)
        network_status.append((0., _id, nstats))
    return network_status
def takeFirst(elem):
    return elem[0]

def generate_edge_from_kf(last_kf, kf, use_bag_cov):
    q0_inv = quaternion_inverse(last_kf.quat)
    _quat = quaternion_multiply(q0_inv, kf.quat)
    _pos = kf.pos - last_kf.pos
    l = np.linalg.norm(_pos)
    if l < 0.05:
        l = 0.05
    if use_bag_cov:
        inf_mat = 1.0/np.array([l*VO_POS_COV_PER_METER, l*VO_POS_COV_PER_METER, l*VO_POS_COV_PER_METER, 
            l*VO_ATT_COV_PER_METER, l*VO_ATT_COV_PER_METER, l*VO_ATT_COV_PER_METER])
        inf_mat = np.diag(inf_mat)
    else:
        inf_mat = np.identity(6)
    _pos = quaternion_matrix(q0_inv)[0:3, 0:3]@_pos
    edge = Edge(last_kf.keyframe_id, kf.keyframe_id, _pos, _quat, False, inf_mat=inf_mat)
    return edge

def readbag(bagpath, extra_bagpathes = [], use_rawid=False, good_comm_start = True, start_id =1, use_bag_cov=False):
    bag = rosbag.Bag(bagpath)
    kfs = {}
    all_kf = {}
    kf_seq = []
    ts = {}
    t0 = None
    pg = PoseGraph()
    kf_count = 0
    kf_index = {}
    drone_ids = set()
    for topic, msg, t in bag.read_messages(topics=['/swarm_loop/keyframe']):
        if t0 is None:
            t0 = msg.header.stamp.to_sec()
        drone_id = msg.drone_id - start_id
        drone_ids.add(drone_id)
        if drone_id not in kfs:
            kfs[drone_id] = []
            
        _pos = np.array([msg.position.x, msg.position.y, msg.position.z])
        _quat = np.array([msg.quat.w, msg.quat.x, msg.quat.y, msg.quat.z])
        if use_rawid:
            kf = KeyFrame(msg.keyframe_id, drone_id,  _pos, _quat, drone_id=drone_id)
            ts[msg.keyframe_id] = msg.header.stamp.to_sec() - t0
        else:
            kf = KeyFrame(kf_count, drone_id,  _pos, _quat, drone_id=drone_id)
            ts[kf_count] = msg.header.stamp.to_sec() - t0

        all_kf[msg.keyframe_id] = kf
        kf_index[msg.keyframe_id] = kf_count

        if len(kfs[drone_id]) > 0:
            last_kf = kfs[drone_id][-1]
            edge = generate_edge_from_kf(last_kf, kf, use_bag_cov)
            kf.add_edge(edge)

        kfs[drone_id].append(kf)
        kf_count += 1

    edge_count = 0
    for topic, msg, t in bag.read_messages(topics=['/swarm_loop/loop_connection']):
        _pos = np.array([msg.relative_pose.position.x, msg.relative_pose.position.y, msg.relative_pose.position.z])
        _quat = np.array([msg.relative_pose.orientation.w, msg.relative_pose.orientation.x, 
                        msg.relative_pose.orientation.y, msg.relative_pose.orientation.z])
        is_inter = msg.drone_id_a != msg.drone_id_b
        if use_bag_cov:
            inf_mat = 1.0/np.array([msg.pos_cov.x, msg.pos_cov.y, msg.pos_cov.z, 
                    msg.ang_cov.x, msg.ang_cov.y, msg.ang_cov.z])
            inf_mat = np.diag(inf_mat)
        else:
            inf_mat = np.identity(6)
        if use_rawid:
            edge = Edge(msg.keyframe_id_a, msg.keyframe_id_b, _pos, _quat, is_inter, inf_mat=inf_mat)
        else:
            edge = Edge(kf_index[msg.keyframe_id_a], kf_index[msg.keyframe_id_b],
                _pos, _quat, is_inter, inf_mat=inf_mat)
        edge_count+= 1
        if msg.keyframe_id_a in all_kf and msg.keyframe_id_b in all_kf:
            all_kf[msg.keyframe_id_a].add_edge(edge)
            if is_inter:
                all_kf[msg.keyframe_id_b].add_edge(edge)

    for kf_id in all_kf:
        kf = all_kf[kf_id]
        kf_seq.append((ts[kf.keyframe_id], kf))
        pg.add_keyframe(kf, add_edges_from_keyframe = True, agent_id=0)

    print("agents", list(kfs.keys()))
    for _id in kfs:
        print(f"Drone {_id}: {len(kfs[_id])} keyframes")
    print("edges", edge_count)

    if good_comm_start:
        network_status = create_connected_network_statues(drone_ids)
    network_status = network_status + read_network_status(1, bag, t0, start_id)
    for _it in extra_bagpathes:
        network_status = network_status + read_network_status(_it[0], 
            rosbag.Bag(_it[1]), t0, start_id)
    network_status.sort(key=takeFirst)
    return kf_seq, list(kfs.keys()), network_status, pg
    

def process_data(bagpath, data_folder, extra_bagpaths = [], comm_mode="router", show=False, verbose=False, 
        solve_duration = 10, repart_keyframe = 100, no_dpgo=False, show_on_end=False, load_result=False):
    comm_range = 100000
    pathes, agents, network_status, pg = readbag(bagpath, extra_bagpaths)
    if verbose:
        print("Fininsh read bag\nGenerating Groundtruth...")
    pg_gt_dgs = generate_groundtruth_DGS(pg, data_folder, agent_num=1)
    pg_gt_sesync = generate_groundtruth_sesync(pg, data_folder)
    pg_gt = pg_gt_sesync
    
    if verbose:
        ate_p, ate_q = ATE(pg, pg_gt)
        print(f"Raw ATE {ate_p:.3f} {ate_q*57.3:.1f}deg")

    if show or True:
        ate_p, ate_q = ATE(pg, pg_gt)
        ate_p_gt, ate_q_gt = ATE(pg_gt_sesync, pg_gt_dgs)
        print(f"Raw ATE {ate_p:.3f} {ate_q*57.3:.1f}deg SE-sync/DGS {ate_p_gt} {ate_q_gt*57.3}")
        
        pg_show = copy.deepcopy(pg)
        _pg_gt_dgs = copy.deepcopy(pg_gt_dgs)
        align_posegraphs(_pg_gt_dgs, pg_show)
        align_posegraphs(pg_gt_sesync, pg_show)
        ax = pg_show.show("raw", plot3d=False, show_edges=False)
        _pg_gt_dgs.show("DGS", ax=ax, clear=False, plot3d=False, color_override="orange", show_edges=False)
        pg_gt_sesync.show("SESYNC", ax=ax, clear=False, plot3d=False, color_override="green", show_edges=False)
    
    _pathes = copy.deepcopy(pathes)
    _pg_gt = copy.deepcopy(pg_gt)
    DSLAM_PATH = f"{data_folder}/DSLAM-baseline/"
    pgms = simulate_realtime_multiagents_slam2_kf_sequency(_pathes, _pg_gt, agents, use_greedy=False, use_repart=False, 
            repart_keyframe=repart_keyframe, solve_duration=solve_duration, comm_mode=comm_mode, comm_range=comm_range, 
            show_on_end=show_on_end, network_status=network_status, dslam_path=DSLAM_PATH, show=show,verbose=verbose, 
            title="Baseline", no_dpgo=no_dpgo, load_result=load_result)

    _pathes = copy.deepcopy(pathes)
    _pg_gt = copy.deepcopy(pg_gt)
    DSLAM_PATH = f"{data_folder}/DSLAM-nogreedy/"
    pgms = simulate_realtime_multiagents_slam2_kf_sequency(_pathes, _pg_gt, agents, use_greedy=False, use_repart=True, 
        repart_keyframe=repart_keyframe, solve_duration=solve_duration, comm_mode=comm_mode, comm_range=comm_range, 
        show_on_end=show_on_end, network_status=network_status, dslam_path=DSLAM_PATH, show=show,verbose=verbose, 
        title="NoGreedy", no_dpgo=no_dpgo, load_result=load_result)

    _pathes = copy.deepcopy(pathes)
    _pg_gt = copy.deepcopy(pg_gt)
    DSLAM_PATH = f"{data_folder}/DSLAM-proposed/"
    pgms = simulate_realtime_multiagents_slam2_kf_sequency(_pathes, _pg_gt, agents, use_greedy=True, use_repart=True, 
            repart_keyframe=repart_keyframe, solve_duration=solve_duration, comm_mode=comm_mode, comm_range=comm_range, 
            show_on_end=show_on_end, network_status=network_status, dslam_path=DSLAM_PATH, show=show,verbose=verbose, 
            title="Proposed", no_dpgo=no_dpgo, load_result=load_result)
    
if __name__ == "__main__":
    master_bagpath = "/home/xuhao/data/ral2021/raw/drone3-RI/outputs/fuse_all/swarm1/swarm_local_pc.bag"
    extra_bagpaths = [
        (2, "/home/xuhao/data/ral2021/raw/drone3-RI/outputs/fuse_all/swarm2/swarm_local_pc.bag"),
        (3, "/home/xuhao/data/ral2021/raw/drone3-RI/outputs/fuse_all/swarm3/swarm_local_pc.bag")
    ] #For comm

    process_data(master_bagpath, "/home/xuhao/data/ral2021/DGS/realworld-drone3-comm/", extra_bagpaths=extra_bagpaths, 
        comm_mode="bag", show=False, show_on_end=True, verbose=True, solve_duration=100, load_result=True)
    # process_data(master_bagpath, "/home/xuhao/data/ral2021/DGS/realworld-drone3/", comm_mode="router", show=False, verbose=False,
    #         solve_duration=100, load_result=True, show_on_end=True)
    plt.show()