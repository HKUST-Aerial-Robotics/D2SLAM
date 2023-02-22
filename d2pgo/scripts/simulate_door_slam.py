#!/usr/bin/env python
from pose_graph_partitioning.pose_graph import *
from simulate_bdslam_realworld import *

def read_pgo(path, iter_idx, nodes, max_iters):
    pgo = PoseGraph()
    for i in nodes:
        idx = i-1
        pgo.agents[idx] = Agent(idx) #Note that the agent id starts from 0
        if max_iters is not None:
            _iter = min(max_iters[i], iter_idx)
        else:
            _iter = iter_idx
        # print(f"{path}/swarm{i}/g2o_drone_{i}_iter_{_iter}.g2o")
        pgo.read_g2o(f"{path}/swarm{i}/g2o_drone_{i}_iter_{_iter}.g2o", idx)
    # Read the frame_id to stamp mapping from frame_timestamp.txt
    frame_id_to_stamp = {}
    for i in nodes:
        with open(f"{path}/swarm{i}/frame_timestamp.txt", "r") as f:
            for line in f:
                frame_id, stamp = line.split()
                frame_id_to_stamp[int(frame_id)] = float(stamp)
    pgo.update_edges()
    return pgo, frame_id_to_stamp

def solve_pgo(path, iter_idx, nodes, working_folder, pgo_old=None, thres=0.01, max_iters=None):
    pgo_input, frame_id_to_stamp = read_pgo(path, iter_idx, nodes, max_iters)
    output_path =  working_folder + f"/time{iter_idx}/"
    pgo_input.write_to_g2o_folder(output_path, cvt_id=True)
    try:
        iterations, _, max_time, initial, final, _, total_optim = call_DGS_solver(output_path, len(nodes), rthresh=thres, pthresh=thres, maxIter=100)
        volume = pgo_input.communication_volume(broadcast=True) * iterations
        pgo_optimized = copy.copy(pgo_input)
        pgo_optimized.read_g2o_single(f"{output_path}/fullGraph_optimized.g2o", update_only=True)
        print(f"iter: {iter_idx}, iterations: {iterations}, max_time: {max_time}, total_time {total_optim}, initial: {initial}, final: {final}, volume: {volume} (poses)")
    except:
        # print(f"Failed solve DGS")
        return None, None, None, None, None
    for i in nodes:
        pgo_optimized.agents[i-1].write_to_csv(f"{working_folder}/pgo_{i}.csv", frame_id_to_stamp)

    pgo_latest_states = {i:[] for i in nodes}
    frame_id0 = min(pgo_input.agents[0].get_keyframe_ids())
    pos0, quat0 = pgo_optimized.keyframes[frame_id0].pos, pgo_input.keyframes[frame_id0].quat
    R0 = quaternion_matrix(quat0)[0:3,0:3]
    c = 0
    for agent_id in pgo_input.agents:
        # Find the keyframe with largest frame_id
        robot_idx = agent_id + 1
        agent = pgo_input.agents[agent_id]
        keyframe_ids = agent.get_keyframe_ids()
        for frame_id in keyframe_ids:
            if pgo_old is None or frame_id not in pgo_old.keyframes:
                ts = frame_id_to_stamp[frame_id]
                frame = pgo_optimized.keyframes[frame_id]
                pos, quat = frame.pos, frame.quat
                # Inverse transform by the first keyframe
                # This is for RE gen
                pos = np.matmul(R0.T, pos - pos0)
                quat = quaternion_multiply(quaternion_inverse(quat0), quat)
                pgo_latest_states[robot_idx].append(np.concatenate(([ts], pos, quat)))
                c += 1
        pgo_latest_states[robot_idx] = np.array(pgo_latest_states[robot_idx])
    print("Total number of new states: ", c)
    return max_time, volume, iterations, pgo_latest_states, pgo_optimized

def evaluate_door_slam(path, nodes=[], thres=0.01, step=1):
    working_folder = path + "/door-slam/"
    min_iter = 10000000
    max_iter = 0
    max_iters = {}
    for i in nodes:
        # We need to find the minimum iteration and maximum iteration using file name
        # Sample filename:
        # g2o_drone_1_iter_444.g2o 444:iteration
        # Read filenames
        filenames = os.listdir(f"{path}/swarm{i}")
        # Filter out the filenames that do not contain "g2o_drone" to find the maximum iteration
        filenames = [f for f in filenames if "g2o_drone" in f]
        # Extract the iteration number
        iterations = [int(f.split("_")[-1].split(".")[0]) for f in filenames]
        # Find the minimum and maximum iteration
        _max_iter = max(iterations)
        max_iter = max(_max_iter, max_iter)
        min_iter = min(_max_iter, min_iter)
        max_iters[i] = _max_iter
    print(f"Data min_iter: {min_iter}, max_iter: {max_iter}")
    # Now we can iterate through the iterations
    pgo_latest_states = {i: [] for i in nodes}
    pgo_optim = None
    volume = 0
    solve_time = 0
    count = 0
    iterations = 0
    for iter_idx in range(200, max_iter, step):
        max_time, _volume, iters, _pgo_latest_states, pgo_optim = solve_pgo(path, iter_idx, nodes, working_folder, pgo_optim, 
                                                                        thres=thres, max_iters=max_iters)
        if _pgo_latest_states is not None:
            volume += _volume
            solve_time += max_time
            iterations += iters
            for i in nodes:
                if len(pgo_latest_states[i]) > 0:
                    if len(_pgo_latest_states[i]) > 0:
                        pgo_latest_states[i] = np.concatenate([pgo_latest_states[i], _pgo_latest_states[i]], axis=0)
                else:
                    pgo_latest_states[i] = _pgo_latest_states[i]
            count += 1
    pgo_latest_states = {i: np.array(pgo_latest_states[i]) for i in nodes}
    # Save pgo_latest_states to CSVs
    print(f"Total solve time: {solve_time}, average {solve_time/count} avg iter {iterations/count}, total volume: {volume} (poses)")
    for i in nodes:
        np.savetxt(f"{working_folder}/realtime_{i}.csv", pgo_latest_states[i], delimiter=" ")
    return pgo_latest_states

if __name__ == "__main__":
    path = "/home/xuhao/data/d2slam/quadcam_7inch_n3_2023_1_14/outputs/doorslam-5-yaw"
    nodes = [1, 2, 3, 4, 5]
    solve_pgo(path, 100, nodes)