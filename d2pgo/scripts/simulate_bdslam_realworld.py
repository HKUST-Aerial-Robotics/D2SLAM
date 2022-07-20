from pose_graph_partitioning.pose_graph import *
from pose_graph_partitioning.pose_graph_partitioning import *
from pose_graph_partitioning.tsp_dataset_generation import *
from pose_graph_partitioning.decentralized_graph_partition import *
from simulate_utils import *

import copy
import time

import faulthandler
faulthandler.enable()

DSLAM_PATH = ""

optimizer_path = "/home/xuhao/source/distributed-mapper/distributed_mapper_core/cpp/build/runDistributedMapper"
iters = []
min_times = []
max_times = []
initials = [] 
finals = [] 
totalvs = [] 
aedges = [] 
ut_rates = []
ate_ps = []
ate_qs = []

ax_optimized = None

def reset_logs():
    global iters, min_times, max_times, initials, finals, totalvs, aedges, ut_rates
    iters, min_times, max_times, initials, finals, totalvs, aedges,ut_rates = [], [], [], [], [], [], [], []
    global ate_ps, ate_qs
    ate_ps = []
    ate_qs = []

def print_log(title, pgms, cuts, vols, unbs, time_cost_sum, keyframe_num, pg_gt, master_id=0):
    if title == "Baseline":
        total_v = np.sum(totalvs)
    else:
        total_v = np.sum(totalvs) + count_total_v(pgms)
    ate_p_m, ate_q_m = np.mean(ate_ps), np.mean(ate_qs)
    if len(ate_ps) > 0:
        ate_p, ate_q = ate_ps[-1], ate_qs[-1]
    else:
        ate_p, ate_q = 0., 0.
    print(f"{title} KFs {keyframe_num} cut {np.mean(cuts):.2f} comm_vol {np.mean(vols):.3f} \
unbs {np.mean(unbs):.2f} time {time_cost_sum/keyframe_num*1000:.1f}ms")

    print(f"addition_edges {np.mean(aedges):2.1f} max_time {np.mean(max_times):.1f}ms min_time \
{np.mean(min_times):.1f}ms util_rates {np.mean(ut_rates)*100:.1f}% iterations {np.mean(iters):.1f} initial \
{np.mean(initials):.2f} final {np.mean(finals):.2f} ATE mean {ate_p_m:.3f} {ate_q_m*57.3:.3f} final {ate_p:.3f} {ate_q*57.3:.3f} \
total_v {total_v/MB:.3f}MB overhead { count_total_v(pgms)/MB:.3f}MB")


def solve_callback(ts, pgms, network, duplicate_inter_edge=True,allow_complementary_internal=True, dslam_path="", 
        show=False, pg_gt=None, load_result=False, master_id=0, title=""):
    pgms[master_id].pg.update_edges()
    cut, vol, min_keyframes, max_keyframes,  kf_num, edge_num = pgms[master_id].pg.statistical()
    # print(f"cut {cut} vol {vol} min_keyframes {min_keyframes} max_keyframes {max_keyframes}  kf_num {kf_num} edge_num {edge_num}")
    path = dslam_path + f"/time{ts}/"
    addition_edges = write_to_distributed_solver_cluster(pgms, path, 
            duplicate_inter_edge=duplicate_inter_edge, 
            allow_complementary_internal=allow_complementary_internal, cvt_id=True)
    iterations, min_time, max_time, initial, final,util_rate = call_DGS_solver(path, len(pgms), rthresh=0.1, pthresh=0.1, maxIter=100)

    pg2 = PoseGraph()
    pg2.read_g2o_single(f"{path}/fullGraph_optimized.g2o")
    ate_p, ate_q = ATE(pg2, pg_gt)
    if show:
        global ax_optimized
        ax_optimized = pg2.show(f"optimized", ax=ax_optimized)

    if load_result:
        for i in pgms:
            pgms[i].pg.read_g2o_single(f"{path}/fullGraph_optimized.g2o", update_only=True)
            # print("Loading result... to ", i)
            # plt.figure("debug")
            # ate_p, ate_q = ATE(pgms[i].pg, pg_gt, debug_title="pgms[i].pg, pg_gt", debug=True)
            # ate_p_, ate_q_ = ATE(pg2, pg_gt, debug_title="pg2, pg_gt", debug=True)
            # plt.grid()
            # plt.legend()
            # plt.pause(1.0)
            # print("pg2", len(pg2.keyframes), len(pgms[i].pg.keyframes), "ate", ate_p, ate_q, ate_p_, ate_q_)

    ###FOR DEBUG
    # if load_result:
    #     pg2_show = copy.deepcopy(pg2)
    #     pggt_show = copy.deepcopy(pg_gt)
    #     ate_p, ate_q = ATE(pgms[master_id].pg, pg_gt)
    #     align_posegraphs(pg2_show, pg_gt)
    #     print("keyframes", len(pgms[master_id].pg.keyframes), "pg2", len(pg2.keyframes),"Ate_p", ate_p, "Ate_q", ate_q)
    #     print("final cost", final)

    #     ax = pg2_show.show(f"{title} optimized DEBUG", plot3d=False, show_edges=False)
    #     pggt_show.show(f"{title} GT DEBUG", ax=ax, clear=False, plot3d=False, 
    #             color_override="red", show_edges=False)
    ##

    iters.append(iterations)
    min_times.append(min_time)
    max_times.append(max_time)
    initials.append(initial)
    finals.append(final)
    totalvs.append(vol*iterations)
    aedges.append(addition_edges)
    ut_rates.append(util_rate)
    ate_ps.append(ate_p)
    ate_qs.append(ate_q)

def solve_callback_comm(ts, pgms, network,duplicate_inter_edge=True,
        allow_complementary_internal=True, dslam_path="", show=False, pg_gt=None, load_result=False):
    cut, vol, min_keyframes, max_keyframes,  kf_num, edge_num = pgms[1].pg.statistical()
    for main_id in network.clusters:
        try:
            _pgms = {}
            # print("\n", main_id, "solve", network.clusters[main_id])
            _path =  dslam_path + f"/time{ts}/subswarm_{main_id}/"
            for i in network.clusters[main_id]:
                _pgms[i] = pgms[i]
            addition_edges = write_to_distributed_solver_cluster(_pgms, _path, 
                    duplicate_inter_edge=duplicate_inter_edge, allow_complementary_internal=allow_complementary_internal, cvt_id=True)
            iterations, min_time, max_time, initial, final,util_rate = call_DGS_solver(_path, len(_pgms), rthresh=0.1, pthresh=0.1, maxIter=100)
            if iterations < 0:
                return
            if load_result:
                for i in pgms:
                    pgms[i].pg.read_g2o_single(f"{_path}/fullGraph_optimized.g2o", update_only=True)
            if show:
                pg2 = PoseGraph()
                pg2.read_g2o_single(f"{_path}/fullGraph_optimized.g2o")
                global ax_optimized
                ax_optimized = pg2.show("optimized", ax=ax_optimized)
            iters.append(iterations)
            min_times.append(min_time)
            max_times.append(max_time)
            initials.append(initial)
            finals.append(final)
            totalvs.append(vol*iterations)
            aedges.append(addition_edges)
            ut_rates.append(util_rate)
        except:
            raise

def compuate_network_topology_realworld(agents, swarm_network_statues):
    # print(swarm_network_statues)
    direct_connections = []
    for _id in swarm_network_statues:
        network_status_it = swarm_network_statues[_id]
        self_id = network_status_it[1]
        status = network_status_it[2]
        for drone_conn in status.network_status:
            _id = drone_conn.drone_id
            if drone_conn.active:
                direct_connections.append((self_id, _id))
    # print("conns", direct_connections)
    network_topology = NetworkTopology(agents, direct_connections, "mesh")
    return network_topology

def simulate_realtime_agent_slam_step(pgm: PoseGraphManager, pgms, kf, network_topology:NetworkTopology, 
        use_greedy, use_repart,force_repartition):
    #Then push out path
    agent_id = pgm.self_id

    pgm.set_network_topology(network_topology)

    #pull poses if network_changed
    if pgm.network_topology_changed and use_repart:
        # print(f"network changed to ", pgm.network_topology.clusters)
        for _agent_id in pgm.network_topology.connected_agents(agent_id):
            my_keys = pgm.available_keyframe_ids()
            remote_keys = pgms[_agent_id].available_keyframe_ids()
            diff_keys = remote_keys - my_keys
            addition_kfs = pgms[_agent_id].get_keyframe_copy_ids(diff_keys)
            if len(addition_kfs) > 0:
                pgm.sync_remote_poses(addition_kfs)

    if kf is not None:
        part_id = None
        if not use_greedy:
            part_id = agent_id
        part_id = pgm.add_keyframe(kf, part_id=part_id)
        #Then we need to sync this part to all connected keyframes
        for _agent_id in pgm.network_topology.connected_agents(agent_id):
            _kf = kf.copy()
            pgms[_agent_id].add_keyframe(_kf, part_id=part_id)
    
    reparted = False
    if use_repart:
        id_parts, _agent_num = pgm.adaptive_repart_routine(force_repartition)
        #Sync poses from all connected drones
        if id_parts is not None:
            agent_list = pgm.get_available_agents()
            my_keys =  pgm.available_keyframe_ids()
            for _agent_id in pgm.network_topology.connected_agents(agent_id):
                remote_keys = pgms[_agent_id].available_keyframe_ids()
                diff_keys = my_keys - remote_keys
                addition_kfs = pgm.get_keyframe_copy_ids(diff_keys)
                pgms[_agent_id].sync_full_partitions(_agent_num, id_parts, agent_list, addition_kfs)
            reparted = True
    return reparted


def simulate_realtime_multiagents_slam_step(_ts, pg_gt, keyframe, pgms, cur_positions, comm_mode, comm_range, 
        use_greedy, use_repart, last_network_topology, show=False, solve=False, force_repartition=False, dslam_path="", 
        load_result=False, title="", master_id=0):
    #Timestamps is the stamp of the keyframes
    #Now we iteration over timestamps
    agent_num = len(pgms)
    time_cost = 0

    network_topology = last_network_topology
    agent_id = keyframe.drone_id
    start = time.time()

    simulate_realtime_agent_slam_step(pgms[agent_id], pgms, keyframe, network_topology, 
            use_greedy, use_repart, force_repartition=force_repartition)

    for _id in pgms:
        if _id != agent_id:
            simulate_realtime_agent_slam_step(pgms[_id], pgms, None, network_topology, 
                use_greedy, use_repart, force_repartition=force_repartition)
            
    time_cost += time.time() - start

    # if solve or reparted:
    if solve :
        if comm_mode=="router":
            solve_callback(_ts, pgms, network_topology, dslam_path=dslam_path, show=show, pg_gt=pg_gt, 
                load_result=load_result, title=title, master_id=master_id)
        else:
            solve_callback_comm(_ts, pgms, network_topology, dslam_path=dslam_path, show=show, pg_gt=pg_gt, 
                load_result=load_result)
        
    #We only count first agent now.
    cut, vol = 0, 0
    min_keyframes, max_keyframes = 1e8, -1
    # for _master in network_topology.clusters:
    _master = 1
    _cut, _vol, _min_keyframes, _max_keyframes, kf_num, edge_num = pgms[_master].statistical()
    cut = _cut + cut
    vol = _vol + vol
    if _min_keyframes < min_keyframes:
        min_keyframes = _min_keyframes
    if _max_keyframes > max_keyframes:
        max_keyframes = _max_keyframes
    
    return cut/kf_num, vol/kf_num, max_keyframes, min_keyframes, len(network_topology.clusters), time_cost

def simulate_realtime_multiagents_slam2_kf_sequency(kf_seq, pg_gt, agent_ids, use_greedy=True, use_repart=True, 
        repart_keyframe=100, solve_duration = 100, comm_mode="router", comm_range=1000000, 
        network_status=[],dslam_path="", show=False, verbose=False, title="basic", no_dpgo=False, show_on_end=False, 
        load_result=False, master_id=0):
    #The kf_seq should be sort by ts
    #Commuincation mode:
    #router: all communincation is constant and stable.
    #adhoc: communicate with only neighbors.
    #mesh: communication on mesh network.

    #First of all, our pathes has no timestamps
    #so we need to create some timestamps for it.
    max_t = kf_seq[-1][0]
    reset_logs()
    #GreedyPartitioning Initialization
    pgms = {}
    for i in agent_ids:
        pgms[i] = PoseGraphManager(i, repart_keyframe, agent_ids=agent_ids)
        pgms[i].update_greedy_partition_routine(
                    repart_keyframe)
        # print("New pgm", i)

    network_topology = NetworkTopology(agent_ids,network_mode=comm_mode)
    
    cuts = []
    vols = []
    unbs = []
    cur_positions = {}
    network_clusters = []
    ax = {}
    cul_num_keyframes = 0
    force_repartition = False

    for _id in agent_ids:
        ax[_id] = None

    time_cost_sum = 0
    _ind_net = 0
    network_statues = {}
    for _ts, kf in kf_seq:
        cul_num_keyframes += 1
        end_kf = cul_num_keyframes == len(kf_seq)
        while comm_mode=="bag" and _ind_net < len(network_status) and network_status[_ind_net][0] < _ts:
            _self_id = network_status[_ind_net][1]
            network_statues[_self_id] = network_status[_ind_net]
            _ind_net += 1
        if comm_mode=="bag":
            network_topology = compuate_network_topology_realworld(agent_ids, network_statues)
        need_solve = (cul_num_keyframes%solve_duration) == solve_duration-1
        need_solve = need_solve or end_kf
        need_solve = need_solve and (not no_dpgo)
        force_repartition = end_kf
        _load_result = load_result and end_kf

        cut, vol, max_keyframes, min_keyframes, num_clusters, time_cost = \
                simulate_realtime_multiagents_slam_step(_ts, pg_gt, kf, pgms, cur_positions, comm_mode, comm_range, 
                    use_greedy, use_repart, network_topology, show=show, solve=need_solve, dslam_path=dslam_path,
                    force_repartition=force_repartition, load_result=_load_result, title=title, master_id=master_id)
        cuts.append(cut)
        vols.append(vol)
        if min_keyframes == 0:
            pass #unbs.append(10000)
        else:
            unbs.append(max_keyframes/min_keyframes)
        network_clusters.append(num_clusters)
        if cul_num_keyframes %100 == 0 and verbose:
            print(f"({_ts:05.1f}) KF: {cul_num_keyframes}/{pgms[master_id].keyframe_num()} comm_cluster {num_clusters} \
min_max {min_keyframes}<->{max_keyframes} cut {cut:.5f} comm_vol {vol:.5f} edges {pgms[master_id].edge_num()}")
        if show and cul_num_keyframes%100 == 0:
            for _id in pgms:
                ax[_id] = pgms[_id].show(f"{title} Agent {cul_num_keyframes} {_id}", ax[_id], plot3d=False)
                plt.pause(0.1)
        time_cost_sum += time_cost

    pgms[master_id].pg.update_edges()
    cut, vol, min_keyframes, max_keyframes,  kf_num, edge_num = pgms[master_id].pg.statistical()
    if verbose:
        print(f"Partition agents {len(pgms)} .... cut {cut} vol {vol}  edge_num{edge_num}  unb {unbs[-1]:.2f},  kf_num{kf_num}")
    if pgms[master_id].count_repartitioning > 0:
        avg_time_repart = pgms[master_id].time_cost_repartitioning/pgms[master_id].count_repartitioning
    else:
        avg_time_repart = -1
    print_log(title, pgms, cuts, vols, unbs, time_cost_sum, cul_num_keyframes, pg_gt, master_id)

    _pg_gt = copy.deepcopy(pg_gt)
    if show_on_end:
        for _id in pgms:
            pg_show = copy.deepcopy(pgms[_id].pg)
            align_posegraphs(pg_show, _pg_gt)
            ax = pg_show.show(f"Final_{title}_Agent_{_id}", plot3d=False, show_edges=False, show_title=False)
            # _pg_gt.show(f"Final {title} GT {_id}", ax, clear=False, plot3d=False, show_edges=False,
            #         color_override="red")
            if verbose:
                ate_p, ate_q = ATE(pg_show, _pg_gt)
                print(f"Final {title} GT {_id} ATE {ate_p:.3f} ate_q {ate_q*57.3:.1f}deg")

    return pgms