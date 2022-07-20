
from pose_graph_partitioning.pose_graph import *
from pose_graph_partitioning.tsp_dataset_generation import *
from pose_graph_partitioning.pose_graph_partitioning import *
from simulate_bdslam import *
from simulate_utils import *
import pickle
import os

sesync_bin="/home/xuhao/source/SESync/C++/build/bin/SE-Sync"

def initialize_poses(pg0, data_folder, agent_num, initial_partition, max_iter_initial_n, show=False):
    pg = copy.deepcopy(pg0)
    title = data_folder.split("/")[-1]
    #Here we run DGS for few iteration to initialize
    print("computing inital noised,")
    partitioning(pg, "vol", agent_num=agent_num, show=False, method=initial_partition)
    pg.update_edges()
    pg.write_to_g2o_folder(f"{data_folder}/parted-initial-noise/", cvt_id=True)
    #call optimizer to optimize the data here
    call_DGS_solver(f"{data_folder}/parted-initial-noise", agent_num=agent_num, rthresh=1e-2, pthresh=1e-2, maxIter=max_iter_initial_n)

    pg2 = copy.deepcopy(pg0)
    pg2.read_g2o_single(f"{data_folder}/parted-initial-noise/fullGraph_optimized.g2o", update_only=True)
    print("Loaded initial noised")
    if show:
        pg2.show(title+"_initial_noised")
    return pg2

def generate_data(data_file, data_folder, agent_num, initial_partition="id", need_initial=False, maxInitIter=1):
    pg = PoseGraph()
    pg.read_g2o_single(data_file)
    title = data_folder.split("/")[-1]
    pg2 = generate_groundtruth_sesync(pg, data_folder)
    pg2.show(f"{title} initial_optimized")
    print("Loaded initial optimized")
    
    # # Generate path from g2o
    # #We load the optimized result of g2o and constructed edge between the disconnected poses, after that we load the initial value again.
    tsp_pathes_cache_file = f"{data_folder}/mTSP-pathes-{agent_num}.npy"
    try:
        pathes = np.load(tsp_pathes_cache_file, allow_pickle=True)
        pathes = generate_path_tsp(pg, agent_num, random_weight=True, pg_optimized=pg2, fix_path=True, align_beginning=True, pathes=pathes)
        print("Loaded pathes from file.....")
    except:
        print(f"No current mTSP pathes found at {tsp_pathes_cache_file}, generating mTSP pathes.....")
        pathes = generate_path_tsp(pg, agent_num, random_weight=True, pg_optimized=pg2, fix_path=True, align_beginning=True)
        np.save(tsp_pathes_cache_file, pathes)    
    
    pg_gt = generate_groundtruth_sesync(pg, data_folder)
    # pg_gt = generate_groundtruth_DGS(pg2, data_folder, agent_num=agent_num)
    ate_p, ate_q = ATE(pg2, pg_gt)
    if need_initial:
        initialize_poses(pg, data_folder, agent_num, "id", maxInitIter)

    print(f"TSP ate_p {ate_p:.10f}, {ate_q*57.3:.10f}")
    return  pg, pg_gt, pathes


def perform_evaluation(pg, pg_gt, pathes, data_folder, solve_duration=10, repart_keyframe=100, comm_range=1000000, 
        comm_mode="router", use_dslam=False, no_dpgo=False):
    _pg = copy.deepcopy(pg)
    
    DSLAM_PATH = f"{data_folder}/DSLAM-baseline/"
    pgms, cuts, vols, unbs, network_clusters,_,_, pg_opti = simulate_realtime_multiagents_slam2(_pg, pg_gt, pathes, use_greedy=False,
        use_repart=False, repart_keyframe=repart_keyframe, solve_duration=solve_duration, dslam_path=DSLAM_PATH,
        comm_range=comm_range, show=False, verbose=False, use_dslam=use_dslam, title="Baseline", no_dpgo=no_dpgo)

    _pg = copy.deepcopy(pg)
    DSLAM_PATH = f"{data_folder}/DSLAM-nogreedy/"
    pgms, cuts, vols, unbs, network_clusters,_,_, pg_opti = simulate_realtime_multiagents_slam2(_pg, pg_gt, pathes, use_greedy=False,
        use_repart=True, repart_keyframe=repart_keyframe, solve_duration=solve_duration, dslam_path=DSLAM_PATH,
        comm_range=comm_range, show=False, verbose=False, use_dslam=use_dslam, title="NoGreedy", no_dpgo=no_dpgo)

    _pg = copy.deepcopy(pg)
    DSLAM_PATH = f"{data_folder}/DSLAM-proposed/"
    pgms, cuts, vols, unbs, network_clusters,_,_, pg_opti = simulate_realtime_multiagents_slam2(_pg, pg_gt, pathes, use_greedy=True,
        use_repart=True, repart_keyframe=repart_keyframe, solve_duration=solve_duration, dslam_path=DSLAM_PATH,
        comm_range=comm_range, show=False, verbose=False, use_dslam=use_dslam, title="Proposed", no_dpgo=no_dpgo)

    if not no_dpgo:
        align_posegraphs(pg_opti, pg_gt)
        ax = pg_opti.show("proposed")
        pg_gt.show("GT", ax=ax, clear=False)

def eval_DGS(g2o_file, data_folder, agent_num=10, solve_duration=10, maxInitIter=1, use_dslam=False, no_dpgo=False):
    pg, pg_gt, pathes = generate_data(g2o_file, data_folder, agent_num, need_initial=use_dslam,
        maxInitIter=maxInitIter)
    perform_evaluation(pg, pg_gt, pathes, data_folder, solve_duration=solve_duration, use_dslam=use_dslam, no_dpgo=no_dpgo)

if __name__ == "__main__":
    eval_DGS("/home/xuhao/data/ral2021/raw/sphere.g2o", 
        "/home/xuhao/data/ral2021/DGS/sphere", 10, solve_duration=100)
    plt.show()
    # eval_DGS("/home/xuhao/data/ral2021/raw/grid3D.g2o",  "/home/xuhao/data/ral2021/DGS/grid3D", 10, solve_duration=1000)
    # plt.show()