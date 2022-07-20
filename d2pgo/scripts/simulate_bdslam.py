from pose_graph_partitioning.pose_graph import *
from pose_graph_partitioning.pose_graph_partitioning import *
from pose_graph_partitioning.tsp_dataset_generation import *
from pose_graph_partitioning.decentralized_graph_partition import *
import copy
import time
from simulate_utils import *
import faulthandler
faulthandler.enable()

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
DSLAM_PATH = ""
ax_optimized = None

def reset_logs():
    global iters, min_times, max_times, initials, finals, totalvs, aedges, ut_rates
    iters, min_times, max_times, initials, finals, totalvs, aedges,ut_rates = [], [], [], [], [], [], [], []
    global ate_ps, ate_qs
    ate_ps = []
    ate_qs = []

def print_log(title, pgms, cuts, vols, unbs, time_cost_sum, keyframe_num):
    if title == "Baseline":
        total_v = np.sum(totalvs)
    else:
        total_v = np.sum(totalvs) + count_total_v(pgms)
    ate_p = np.mean(ate_ps)
    ate_q = np.mean(ate_qs)
    print(f"{title} KFs {keyframe_num} cut {np.mean(cuts):.2f} comm_vol {np.mean(vols):.3f} \
unbs {np.mean(unbs):.2f} time {time_cost_sum/keyframe_num*1000:.1f}ms")

    print(f"addition_edges {np.mean(aedges):2.1f} max_time {np.mean(max_times):.1f}ms min_time \
{np.mean(min_times):.1f}ms util_rates {np.mean(ut_rates)*100:.1f}% iterations {np.mean(iters):.1f} initial \
{np.mean(initials):.2f} final {np.mean(finals):.2f} ATE {ate_p:.3f} {ate_q*57.3:.3f} \
total_v {total_v/MB:.3f}MB overhead { count_total_v(pgms)/MB:.3f}MB")

def solve_callback(ts, pgms, network, duplicate_inter_edge=True,allow_complementary_internal=True, use_dslam=False, pg_gt=None):
    pgms[0].pg.update_edges()
    cut, vol, min_keyframes, max_keyframes,  kf_num, edge_num = pgms[0].pg.statistical()
    path = DSLAM_PATH + f"/time{ts}/"
    output_path =  DSLAM_PATH + f"/time{ts}/output/"
    addition_edges = write_to_distributed_solver_cluster(pgms, path, duplicate_inter_edge=duplicate_inter_edge, 
            allow_complementary_internal=allow_complementary_internal)
    if use_dslam:
        iterations, min_time, max_time, initial, final,util_rate = call_dslam_opti(path, output_path)
    else:
        iterations, min_time, max_time, initial, final,util_rate = call_DGS_solver(path, len(pgms), rthresh=0.1, pthresh=0.1, maxIter=100)

    pg2 = PoseGraph()
    pg2.read_g2o_single(f"{path}/fullGraph_optimized.g2o", cvt_id=True)
    ate_p, ate_q = ATE(pg2, pg_gt)
    
    # align_posegraphs(pg2, pg_gt)
    # ax = pg_gt.show("GT")
    # pg2.show("optimized", ax=ax, clear=False)
    # plt.show()
    
    iters.append(iterations)
    min_times.append(min_time)
    max_times.append(max_time)
    initials.append(initial)
    finals.append(final)
    totalvs.append(vol*iterations*SIZE_POSE_EXCHANGE_ITERATIONS)
    aedges.append(addition_edges)
    ut_rates.append(util_rate)
    ate_ps.append(ate_p)
    ate_qs.append(ate_q)

#     print(f"Optimizing..... cut {cut} vol {vol}  edge_num{edge_num}  unb {max_keyframes/min_keyframes:.2f}, kf_num{kf_num} initial {initial} final {final} iters {iterations} max_time {max_time}")
    return pg2

def simulate_realtime_multiagents_slam(pg_full, pathes, use_greedy=True, use_repart=True, repart_keyframe=30):
    #First of all, our pathes has no timestamps
    #so we need to create some timestamps for it.
    max_t = 0
    timestamps = []
    agent_num = len(pathes)
    
    for path in pathes:
        if len(path) > max_t:
            max_t = len(path)

    for path in pathes:
        #assign timestamps to keyframes,
        #Note ts here is NOT integer
        ts = np.linspace(0, max_t, len(path))
        timestamps.append(ts)

    #Now we iteration over timestamps
    ts_steps = np.zeros(len(pathes), dtype=np.int32)
    
    #GreedyPartitioning Initialization
    pg_new = PoseGraph()
    pg_new.initialize_greedy_partition(agent_num, len(pg_full.keyframes), len(pg_full.edges))
    
    cuts = []
    vols = []
    unbs = []

    for _t in range(max_t):
        #Push timestamp
        # print("Timestamp", _t)
        has_keyframe_added = False
        for j in range(agent_num):
            path = pathes[j]
            ts = timestamps[j]
            while ts[ts_steps[j]] < _t:
                #Then push out path
                _pose_index = path[ts_steps[j]]
                ts_steps[j] += 1
                _id = pg_full.index2id[_pose_index]
                kf = pg_full.keyframes[_id]
                # print(f"Agent {j} ts {ts[ts_steps[j]]} ptr {ts_steps[j]} path pt {_pose_index} pose id {_id} edges {len(kf.edges)}")
                kf = copy.deepcopy(kf)
                has_keyframe_added = True
                if use_greedy:
                    pg_new.add_keyframe(kf, add_edges_from_keyframe=True)
                else:
                    pg_new.add_keyframe(kf, agent_id=j, add_edges_from_keyframe=True)
        
            keyframe_num = len(pg_new.keyframes)
            if has_keyframe_added and keyframe_num % repart_keyframe == 0 and keyframe_num > 0:
                if use_repart:
                    cur_parts = pg_new.current_partition_array()
                    obj, parts = partitioning_with_metis(pg_new.setup_graph(False), agent_num, "vol") #repartitioning_with_parmetis(pg_new, agent_num, cur_parts, "vol", itr=0.0001)
                    if parts is not None:
                        pg_new.repart(agent_num, parts)

                pg_new.greedy_partitioning.param.k = agent_num
                pg_new.greedy_partitioning.param.n = keyframe_num+repart_keyframe
                pg_new.greedy_partitioning.param.m = len(pg_full.edges)*(keyframe_num+repart_keyframe)/keyframe_num
        
        # _, inter_edges = pg_new.update_edges()
        cut, vol, min_keyframes, max_keyframes, _, edge_num = pg_new.statistical()
        if min_keyframes > 0:
            cuts.append(cut/keyframe_num)
            vols.append(vol/keyframe_num)
            unbs.append(max_keyframes/min_keyframes)
            for agent_id in pg.agents:
                if not pg_new.agents[agent_id].check_agent_connection():
                    pg_new.show_graphviz()
                    print(f"check_agent_connection failed on agent {agent_id} cut {cut} edges {edge_num}")
                    pg.show_graphviz()
                    plt.show()
                    exit(-1)

        if _t %100 == 99:
            print(f"{_t} partition agents {agent_num} keyframes {min_keyframes}<->{max_keyframes} inter edges {cut} comm_vol {vol}")

    vol, min_keyframes, max_keyframes = pg_new.statistical()
    print(f" {_t} partition agents {agent_num} keyframes {min_keyframes}<->{max_keyframes} inter edges {cut} comm_vol {vol} total keyframes {len(pg_new.keyframes)}")
    return pg_new, cuts, vols, unbs


def simulate_realtime_agent_slam_step(pgm: PoseGraphManager, pgms, kfs, network_topology:NetworkTopology, use_greedy, use_repart,force_repartition):
    #Then push out path
    agent_id = pgm.self_id

    pgm.set_network_topology(network_topology)

    #pull poses if network_changed
    if pgm.network_topology_changed:
        # print(f"network changed to ", pgm.network_topology.clusters)
        for _agent_id in pgm.network_topology.connected_agents(agent_id):
            my_keys = pgm.available_keyframe_ids()
            remote_keys = pgms[_agent_id].available_keyframe_ids()
            diff_keys = remote_keys - my_keys
            addition_kfs = pgms[_agent_id].get_keyframe_copy_ids(diff_keys)
            if len(addition_kfs) > 0:
                pgm.sync_remote_poses(addition_kfs)

    for kf in kfs:
        part_id = None
        if not use_greedy:
            part_id = agent_id
        part_id = pgm.add_keyframe(kf, part_id=part_id)
        #Then we need to sync this part to all connected keyframes
        for _agent_id in pgm.network_topology.connected_agents(agent_id):
            _kf = kf.copy()
            pgms[_agent_id].add_remote_keyframe(_kf, part_id=part_id)
    
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


def simulate_realtime_multiagents_slam_step(_ts, keyframes, pgms, cur_positions, comm_mode, comm_range, 
    use_greedy, use_repart, last_network_topology, show=False, solve=False, force_repartition=False, use_dslam=False, pg_gt=None, no_dpgo=False):
    #Timestamps is the stamp of the keyframes
    #Now we iteration over timestamps
    agent_num = len(pgms)
    time_cost = 0
    for agent_id in keyframes:
        for kf in keyframes[agent_id]:
            cur_positions[agent_id] = kf.pos

    if comm_mode =="mesh":
        _network_topology = compuate_network_topology(cur_positions, comm_range, comm_mode)
        network_changed = not last_network_topology.equal(_network_topology)
        if network_changed and len(last_network_topology.clusters) == 1 and len(_network_topology.clusters) == 1:
            print(f"network changed anormaly\n{last_network_topology.connected}\n{_network_topology.connected}")
        network_topology = _network_topology
    else:
        network_topology = last_network_topology

    reparted = False
    for i in range(agent_num):
        if i in keyframes:
            kf = keyframes[i]
        else:
            kf = []
        start = time.time()
        _reparted = simulate_realtime_agent_slam_step(pgms[i], pgms, kf, network_topology, use_greedy, use_repart, force_repartition=force_repartition)
        time_cost += time.time() - start
        reparted = reparted or _reparted

    # if solve or reparted:
    pg_optimized = None
    if solve and not(no_dpgo):
        pg_optimized = solve_callback(_ts, pgms, network_topology, use_dslam=use_dslam, pg_gt=pg_gt)
    #We only count first agent now.
    cut, vol = 0, 0
    min_keyframes, max_keyframes = 1e8, -1
    # for _master in network_topology.clusters:
    _master = 0
    _cut, _vol, _min_keyframes, _max_keyframes, kf_num, edge_num = pgms[_master].statistical()
    cut = _cut + cut
    vol = _vol + vol
    if _min_keyframes < min_keyframes:
        min_keyframes = _min_keyframes
    if _max_keyframes > max_keyframes:
        max_keyframes = _max_keyframes
    
    return cut/kf_num, vol/kf_num, max_keyframes, min_keyframes, len(network_topology.clusters), time_cost, pg_optimized

def simulate_realtime_multiagents_slam2(pg_full, pg_gt, pathes, use_greedy=True, use_repart=True, repart_keyframe=100, solve_duration = 100, 
        comm_mode="router", comm_range=1000000, show=False, verbose=False, use_dslam=False, dslam_path="",
        title="basic", no_dpgo=False):
    #Commuincation mode:
    #router: all communincation is constant and stable.
    #adhoc: communicate with only neighbors.
    #mesh: communication on mesh network.
    global DSLAM_PATH
    DSLAM_PATH = dslam_path
    
    reset_logs()
    #First of all, our pathes has no timestamps
    #so we need to create some timestamps for it.
    max_t = 0
    agent_num = len(pathes)
    
    for path in pathes:
        if len(path) > max_t:
            max_t = len(path)

    ts_keyframes = [{} for i in range(max_t)]

    for agent_id in range(agent_num):
        #assign timestamps to keyframes,
        #Note ts here is NOT integer
        path = pathes[agent_id]
        ts = np.linspace(0, max_t-1, len(path))
        for i in range(len(path)):
            _ts = math.floor(ts[i])
            _id = pg_full.index2id[path[i]]
            kf = pg_full.keyframes[_id]
            if agent_id not in ts_keyframes[_ts]:
                ts_keyframes[_ts][agent_id] = []
            ts_keyframes[_ts][agent_id].append(kf)

    #GreedyPartitioning Initialization
    pgms = {}
    for i in range(agent_num):
        pgms[i] = PoseGraphManager(i, repart_keyframe)
        pgms[i].update_greedy_partition_routine(
                    repart_keyframe)

    network_topology = NetworkTopology(range(agent_num),network_mode=comm_mode)
    
    cuts = []
    vols = []
    unbs = []
    cur_positions = {}
    network_clusters = []
    ax = None
    cul_num_keyframes = 0
    force_repartition = False

    time_cost_sum = 0
    pg_opti = None
    for _ts in range(max_t):
        keyframes = ts_keyframes[_ts]
        cul_num_keyframes += len(keyframes)
        need_solve = (_ts%solve_duration) == solve_duration-1
        need_solve = need_solve or _ts == max_t - 1
        # need_solve = _ts == max_t - 1
        force_repartition = _ts == max_t - 1
        cut, vol, max_keyframes, min_keyframes, num_clusters, time_cost, pg_opti_ = \
            simulate_realtime_multiagents_slam_step(_ts, keyframes, pgms, cur_positions, comm_mode, comm_range, 
                use_greedy, use_repart, network_topology, show=show, solve=need_solve, force_repartition=force_repartition,
                use_dslam=use_dslam, pg_gt=pg_gt, no_dpgo=no_dpgo)
        cuts.append(cut)
        vols.append(vol)
        if min_keyframes == 0:
            raise "Error minkeyframe is 0"
        else:
            unbs.append(max_keyframes/min_keyframes)
        network_clusters.append(num_clusters)
        if _ts %100 == 0 and verbose:
            print(f"({_ts:05d}) KF: {cul_num_keyframes}/{pgms[0].keyframe_num()} comm_cluster {num_clusters} min_max {min_keyframes}<->{max_keyframes} cut {cut} comm_vol {vol} edges {pgms[0].edge_num()}")
        if show and _ts%100 == 0:
            ax = pgms[0].show("Agent 0", ax)
            plt.pause(0.01)
        time_cost_sum += time_cost
        if pg_opti_ is not None:
            pg_opti = pg_opti_

    pgms[0].pg.update_edges()
    cut, vol, min_keyframes, max_keyframes,  kf_num, edge_num = pgms[0].pg.statistical()
    print(f"Partition agents {agent_num} .... cut {cut} vol {vol}  edge_num{edge_num}  unb {unbs[-1]:.2f},  kf_num{kf_num}")
    
    if pgms[0].count_repartitioning > 0:
        avg_time_repart = pgms[0].time_cost_repartitioning/pgms[0].count_repartitioning
    else:
        avg_time_repart = -1
    print_log(title, pgms, cuts, vols, unbs, time_cost_sum, cul_num_keyframes)
    return pgms, cuts, vols, unbs, network_clusters, time_cost_sum, avg_time_repart, pg_opti
