from pose_graph_partitioning.pose_graph import *
from pose_graph_partitioning.pose_graph_partitioning import *
from pose_graph_partitioning.tsp_dataset_generation import *
from pose_graph_partitioning.decentralized_graph_partition import *
import copy
import time
import subprocess
from pathlib import Path
import re

    
def compuate_network_topology(cur_positions, comm_range, network_mode):
    agents = []
    direct_connections = []
    for i in cur_positions:
        agents.append(i)
        for j in cur_positions:
            if i != j:
                distance = np.linalg.norm(cur_positions[i] - cur_positions[j])
                if distance < comm_range:
                    direct_connections.append((i, j))
    network_topology = NetworkTopology(agents, direct_connections, network_mode)
    return network_topology

def write_to_distributed_solver_cluster(pgms, path="/home/xuhao/output/DSLAM", debug=False, allow_complementary_internal=True, 
        duplicate_inter_edge=True, cvt_id=True):
    #duplicate_inter_edge True for DGS False for DSLAM
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
    c = 0

    _tmp_id = 0
    force_ids = {}
    for agent_id in pgms:
        force_ids[agent_id] = _tmp_id
        _tmp_id+=1

    _tmp_id = 0
    for agent_id in pgms:
        pgm = pgms[agent_id]
        agent_id = pgm.self_id
        _agent = pgm.pg.agents[agent_id]
        pgm.pg.update_edges(duplicate_inter_edge=duplicate_inter_edge)
        addition_edges = []
        if allow_complementary_internal:
            addition_edges = pgm.pg.ComplementaryAgentInternalConnections(agent_id) 
            if len(addition_edges) > 0:
                c += len(addition_edges)
        agent_id = pgm.self_id
        _agent.write_to_g2o(f"{path}/{_tmp_id}.g2o", cvt_id=cvt_id, addition_edges = addition_edges, force_ids=force_ids)
        _tmp_id+= 1

    return c
    # if debug:
        # pgms[0].pg.show_graphviz()

def write_to_distributed_solver(pgms, network_topology, cvt_id=True):
    #Here we run distributed solver for each clusters in wireless network
    for _master in network_topology.clusters:
        _pgms = {_id: pgms[_id] for _id in network_topology.clusters[_master]}
        # print(f"Running on network cluster {_master}:{network_topology.clusters[_master]}")
        write_to_distributed_solver_cluster(_pgms, cvt_id=cvt_id)

OPTIM_PATH = "/home/xuhao/source/distributed-mapper/distributed_mapper_core/cpp/build/runDistributedMapper"

def call_DGS_solver(path, agent_num, optimizer_path=OPTIM_PATH, rthresh=1e-2, pthresh=1e-2, maxIter=100, between_noise="false"):
    command = f"{optimizer_path}  --nrRobots {agent_num} --dataDir {path}/  --maxIter {maxIter} \
--rthresh {rthresh} --pthresh {pthresh} --useBetweenNoise {between_noise}"
    s = os.popen(command)
    output = s.read()
    # print(output)
    try:
        iterations, min_time, max_time, initial, final, util_rate, total_optim = pocess_DGS_result(output)
        return iterations, min_time, max_time, initial, final, util_rate, total_optim
    except:
        # print("DGS Error!", command, "RET:", output, "\n")
        return -1, 0, 0, 0, 0, 0, 0
        
def call_SESync(path, path_output, optimizer_path="/home/xuhao/source/SESync/C++/build/bin/SE-Sync"):
    command = f"{optimizer_path} {path} {path_output}"
    # print(command)
    s = os.popen(command)
    output = s.read()

def call_dslam_opti(g2o_folder,  output_folder, rate=1e-3, tor=1e-4, is_async="true", silent="true", simulate_delay_ms=20, max_solving_time=1000):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    command = f"roslaunch dslam_pose_graph_opti dpgo_10_g2o.launch g2o_path:={g2o_folder} is_async:={is_async} descent_rate:={rate} tolerance:={tor} silent:={silent} output_folder:={output_folder} simulate_delay_ms:={simulate_delay_ms} max_solve_time:={max_solving_time}"
    s = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    start = time.time()
    while s.poll() is None:
        time.sleep(0.1)
        if time.time() - start > max_solving_time + 5:
            s.kill()
            return call_dslam_opti(g2o_folder,  output_folder, rate=rate, tor=tor, is_async=is_async, silent=silent, 
                                   simulate_delay_ms=simulate_delay_ms, max_solving_time=max_solving_time)
    output = s.stdout.read().decode("utf-8") 
        
    min_it, max_it, initial, final, totalv = pocess_DSLAM_result(output)
    return min_it, max_it, initial, final, totalv, 0.0

def call_d2pgo_opti(g2o_folder,  output_folder, agent_num = 5, ignore_infor = False,
        simulate_delay_ms=0, max_steps=100, enable_rot_init=True, enable_linear_pose6d_solver=False,
        eta_k=1.0, rho_frame_theta=1.0, rho_frame_T=0.25, max_solving_time=10.0, rho_rot_mat=0.09, is_single_mode=False):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    if is_single_mode:
        command = f"roslaunch d2pgo d2pgo_test_single.launch g2o_path:={g2o_folder} \
            output_path:={output_folder} enable_rot_init:={enable_rot_init} ignore_infor:={ignore_infor} \
            enable_linear_pose6d_solver:={enable_linear_pose6d_solver} solver_type:='ceres'"
        # print(command)
    else:
        command = f"roslaunch d2pgo d2pgo_test_multi.launch agent_num:={agent_num} g2o_path:={g2o_folder} \
            output_path:={output_folder} enable_rot_init:={enable_rot_init} max_steps:={max_steps} ignore_infor:={ignore_infor} \
            eta_k:={eta_k} rho_frame_theta:={rho_frame_theta} rho_frame_T:={rho_frame_T} simulate_delay_ms:={simulate_delay_ms} \
            enable_linear_pose6d_solver:={enable_linear_pose6d_solver} debug_rot_init_only:=false \
            rho_rot_mat:={rho_rot_mat} max_solving_time:={max_solving_time}"
        # print(command)
    s = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = s.stdout.read().decode("utf-8")
    err = s.stderr.read()
    pg = PoseGraph()
    pg.read_g2o_folder(output_folder, prt=False)
    # Match solve time from output using regex
    # Sample: [D2PGO1] Solve done. Time: 100.0ms iters 10
    solve_time = []
    iters = []
    datas = re.findall(r"\[D2PGO\d+\] Solve done. Time: (\d+\.\d+)ms iters (\d+)", output)
    for data in datas:
        solve_time.append(float(data[0]))
        iters.append(int(data[1]))
    # Extract RotInit time of each iteration
    # Sample: [RotInit2] RotInit 5.48ms setup 1.30ms LLT 3.94ms Recover 0.24ms state_changes 100.0% Poses 677 EffPoses 677 Loops 709 Priors 5 F32: 1 g_prior: 0
    datas = re.findall(r"\[RotInit\d+\] RotInit (\d+\.\d+)ms", output)
    rot_init_time = np.array([float(data) for data in datas]).astype(float)
    
    # Extract ceres solve time of each iteration
    # [D2PGO::solve@1] solve_count 14 mode [multi,1] total frames 1439 loops 1603 opti_time 36.8ms iters 5 initial cost 6.44e+01 final cost 2.81e+01
    datas = re.findall(r"\[D2PGO::solve@\d+\] solve_count \d+ mode \[multi,\d+\] total frames \d+ loops \d+ opti_time (\d+\.\d+)ms iters (\d+) initial cost", output)
    ceres_solve_time = np.array([float(data[0]) for data in datas]).astype(float)
    ceres_iters = np.array([float(data[1]) for data in datas]).astype(float)
    cere_per_iter = ceres_solve_time / ceres_iters

    ret = {
        "max_solve_time": max(solve_time),
        "mean_iters": np.mean(iters),
        "rot_init_time": rot_init_time,
        "ceres_solve_time": ceres_solve_time,
        "ceres_iters": ceres_iters,
        "cere_per_iter": cere_per_iter
    }
    return pg, ret

def loadSESyncResult(path, pg: PoseGraph):
    data = np.loadtxt(path)
    poses_num = data.shape[1]//4
    assert poses_num == len(pg.keyframes), f"poses num error {poses_num}/{len(pg.keyframes)}"
    for i in range(0, poses_num):
        T = data[:, i]
        R = data[:, poses_num+i*3:poses_num+i*3+3]
        _id = pg.index2id[i]
        pg.keyframes[_id].pos = T
        R_ = np.identity(4)
        R_[0:3, 0:3] = R
        pg.keyframes[_id].quat = quaternion_from_matrix(R_)
    return pg

def loadDPGOResult(path, pg: PoseGraph):
    data = np.loadtxt(path, delimiter=",")
    poses_num = data.shape[1]//4
    assert poses_num == len(pg.keyframes), f"poses num error {poses_num}/{len(pg.keyframes)}"
    for i in range(0, poses_num):
        T = data[:, i]
        R = data[:, poses_num+i*3:poses_num+i*3+3]
        _id = pg.index2id[i]
        pg.keyframes[_id].pos = T
        R_ = np.identity(4)
        R_[0:3, 0:3] = R
        pg.keyframes[_id].quat = quaternion_from_matrix(R_)
    return pg

def angular_error_quat(quat1, quat2):
    dq = quaternion_multiply(quaternion_inverse(quat1), quat2)
    dq = unit_vector(dq)
    if dq[0] < 0:
        dq = - dq
    try:
        angle = 2*acos(dq[0])
    except:
        print("angle err", quat1, quat2, dq[0])
        angle = 0
    return angle

def align_posegraph(pg0):
    _key0 = min(list(pg0.keyframes.keys()))
    pos0_0 = pg0.keyframes[_key0].pos
    quat0_0 = pg0.keyframes[_key0].quat
    invq0_0 = quaternion_inverse(quat0_0)

    for i in pg0.keyframes:
        pos0 = pg0.keyframes[i].pos
        quat0 = pg0.keyframes[i].quat
        dpos0 = pos0 - pos0_0
        dpos0 = quaternion_matrix(invq0_0)[0:3, 0:3]@dpos0
        dquat0 = quaternion_multiply(invq0_0, quat0)
        pg0.keyframes[i].pos = dpos0
        pg0.keyframes[i].quat = dquat0

def align_posegraphs(pg0, gt):
    assert len(pg0.keyframes) <= len(gt.keyframes), f"poses num error {len(pg0.keyframes)}/{len(pg1.keyframes)}"
    _key0 = min(list(gt.keyframes.keys()))
    pos0_0 = pg0.keyframes[_key0].pos
    quat0_0 = pg0.keyframes[_key0].quat
    invq0_0 = quaternion_inverse(quat0_0)
    pos1_0 = gt.keyframes[_key0].pos
    quat1_0 = gt.keyframes[_key0].quat
    invq1_0 = quaternion_inverse(quat1_0)

    for i in pg0.keyframes:
        pos0 = pg0.keyframes[i].pos
        quat0 = pg0.keyframes[i].quat
        dpos0 = pos0 - pos0_0
        dpos0 = quaternion_matrix(invq0_0)[0:3, 0:3]@dpos0
        dquat0 = quaternion_multiply(invq0_0, quat0)
        pg0.keyframes[i].pos = dpos0
        pg0.keyframes[i].quat = dquat0

    for i in gt.keyframes:
        pos1 = gt.keyframes[i].pos
        quat1 = gt.keyframes[i].quat
        dpos1 = pos1 - pos1_0
        dpos1 = quaternion_matrix(invq1_0)[0:3, 0:3]@dpos1
        dquat1 = quaternion_multiply(invq1_0, quat1)
        gt.keyframes[i].pos = dpos1
        gt.keyframes[i].quat = dquat1
     
def ATE(pg0, gt, debug=False, debug_title=""):
    assert len(pg0.keyframes) <= len(gt.keyframes), f"poses num error {len(pg0.keyframes)}/{len(gt.keyframes)}"
    _key0 = min(list(pg0.keyframes.keys()))
    pos0_0 = pg0.keyframes[_key0].pos
    quat0_0 = pg0.keyframes[_key0].quat
    invq0_0 = quaternion_inverse(quat0_0)
    pos1_0 = gt.keyframes[_key0].pos
    quat1_0 = gt.keyframes[_key0].quat
    invq1_0 = quaternion_inverse(quat1_0)

    ate_p_sum = 0
    ate_q_sum = 0

    ate_ps = []
    ate_qs = []

    if debug:
        print("align", pos0_0, quat0_0, pos1_0, quat1_0)

    for i in pg0.keyframes:
        pos0 = pg0.keyframes[i].pos
        quat0 = pg0.keyframes[i].quat
        pos1 = gt.keyframes[i].pos
        quat1 = gt.keyframes[i].quat

        dpos0 = pos0 - pos0_0
        dpos0 = quaternion_matrix(invq0_0)[0:3, 0:3]@dpos0
        dquat0 = quaternion_multiply(invq0_0, quat0)
        dpos1 = pos1 - pos1_0
        dpos1 = quaternion_matrix(invq1_0)[0:3, 0:3]@dpos1
        dquat1 = quaternion_multiply(invq1_0, quat1)
        
        ate_p_sum += np.linalg.norm(dpos1-dpos0)
        ate_q_sum += angular_error_quat(dquat0, dquat1)

        ate_ps.append(np.linalg.norm(dpos1-dpos0))
        ate_qs.append(angular_error_quat(dquat0, dquat1))

    if debug:
        plt.scatter(list(pg0.keyframes.keys()), ate_ps, label=debug_title)

    return ate_p_sum/len(pg0.keyframes), ate_q_sum/len(pg0.keyframes)

def pocess_DGS_result(output):
    import re
    iterations = int(re.findall("\[optimizeRotation\] Finish iteration(\d+)", output, flags=0)[0])
    iterations += int(re.findall("\[optimizePoses\] Finish iteration(\d+)", output, flags=0)[0])
    min_time = float(re.findall("\[DGS\] min ([0-9]+\.?[0-9]*)ms", output, flags=0)[0])
    max_time = float(re.findall("\[DGS\] max ([0-9]+\.?[0-9]*)ms", output, flags=0)[0])
    solve_times = re.findall("\[DGS\] Robot \(\d+\) time (\S+)ms", output, flags=0)
    initial = float(re.findall("Initial error\s+(\S+)", output, flags=0)[0])
    final = float(re.findall("Distributed Error:\s+(\S+)", output, flags=0)[0])
    total_optim = float(re.findall("optimizedPoses time: ([0-9]+\.?[0-9]*)ms", output, flags=0)[0])
    sum_of_solve_time = 0
    for t in solve_times:
        sum_of_solve_time += float(t)
    ut_rate  = sum_of_solve_time/(len(solve_times)*max_time)
    return iterations, min_time, max_time, initial, final, ut_rate, total_optim

def pocess_DSLAM_result(output):
    print(output)
    import re
    _it = re.findall("iteration (\d+)<->(\d+)", output, flags=0)[0]
    min_it = int(_it[0])
    max_it = int(_it[1])
    costs = re.findall("Finished solving final global cost (\S+) / (\S+)", output, flags=0)[0]
    initial = float(costs[1])
    final = float(costs[0])
    totalv = int(re.findall("total_comm_volume (\d+)", output, flags=0)[0])
    return min_it, max_it, initial, final, totalv

def count_total_v(pgms):
    vol = 0
    for i in pgms:
        vol+= pgms[i].total_v
    return vol

MB=1024*1024

def generate_groundtruth_sesync(pg0: PoseGraph, data_folder):
    pg: PoseGraph = copy.deepcopy(pg0)
    if pg.agent_num() != 1:
        partitioning(pg, agent_num=1,method="union")
    pg.rename_keyframes_by_index()
    pg.write_to_g2o_folder(f"{data_folder}/SE-Sync", cvt_id=False, update_edges=False)
    optimized_path = f"{data_folder}/SE-Sync/optimized_poses.txt"
    call_SESync(f"{data_folder}/SE-Sync/0.g2o", optimized_path)
    pg = loadSESyncResult(optimized_path, pg)
    pg.rename_keyframes_from_index()
    return pg

def generate_groundtruth_DGS(pg, data_folder, agent_num=10):
    pg0 = copy.deepcopy(pg)
    partitioning(pg0, agent_num=1,method="union")
    
    pg0.write_to_g2o_folder(f"{data_folder}/groundtruth", cvt_id=False)
    call_DGS_solver(f"{data_folder}/groundtruth", agent_num=1, rthresh=1e-4, pthresh=1e-4, maxIter=100)
    pg2 = PoseGraph()
    pg2.read_g2o_single(f"{data_folder}/groundtruth/fullGraph_optimized.g2o")
    return pg2

