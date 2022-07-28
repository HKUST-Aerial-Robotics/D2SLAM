from pose_graph_partitioning.pose_graph import *

def streaming_graph_partition(pg, agent_num=None, method=FENNEL_PARTITIONING, iteration = 10):
    if agent_num is None:
        agent_num = pg.agent_num
    param = GreedyPartitioningParam(agent_num, len(pg.keyframes), len(pg.edges), gamma=1.5)
    # param.alpha = len(pg.keyframes) / agent_num
    kf_num = len(pg.keyframes)
    parts = np.zeros(kf_num, dtype=np.int)

    keyframes_index = np.linspace(0, kf_num-1, kf_num, dtype=np.int)
    # print(keyframes_index)
    old_partitioning = None
    for iter in range(iteration):
        np.random.shuffle(keyframes_index)
        partitioning = GreedyPartitioning(param=param, method=method)
        for _index in keyframes_index:
            _id = pg.index2id[_index]
            kf = pg.keyframes[_id]
            adjacency_list = []
            for edge in kf.edges:
                if edge.keyframe_ida == _id:
                    adjacency_list.append(edge.keyframe_idb)
                else:
                    adjacency_list.append(edge.keyframe_ida)
            #Processing vertex
            _part_id = partitioning.partition_vertex(_id, adjacency_list, old_partitioning)
            parts[_index] = _part_id
        # param.alpha = param.alpha * 1.1
        old_partitioning = partitioning
    return -1, parts

def partitioning_with_metis(G, agent_num, obj):
    import metis, time
    start = time.time()
    (cut, parts) = metis.part_graph(G, agent_num, objtype=obj) 
    end = time.time()
    # print(f"Metis takes {(end - start)*1000}ms")
    return cut, parts

def partitioning_with_parmetis(pg, agent_num, ubvec=1.05):
    from mgmetis import parmetis
    import time
    xadj, adjncy = pg.cvt_CSR()
    start = time.time()
    ubvec = np.ones(len(xadj)-1)*ubvec
    objval, parts = parmetis.part_kway(agent_num, xadj, adjncy, ubvec=ubvec)
    end = time.time()
    # print(f"ParMetis takes {(end - start)*1000}ms")
    return objval, parts

def repartitioning_with_parmetis(pg, agent_num, parts, obj, itr):
    from mgmetis import parmetis
    import time
    xadj, adjncy = pg.cvt_CSR()
    start = time.time()
    objval, parts = parmetis.adaptive_repart_kway(agent_num, xadj, adjncy, parts, itr=itr)
    end = time.time()
    objval = -1
    # print(f"ParMetis adptive_repart_kway takes {(end - start)*1000}ms")
    return objval, parts

def partitioning(pg:PoseGraph, obj="vol", agent_num=None, show=False, method="METIS", iteration=10, pathes=None, parts=None, itr=1000):
    if agent_num is None:
        agent_num = pg.agent_num

    if method == "METIS":
        # (cut, parts) = pg.partitioning_with_metis(G, agent_num, obj) 
        cut, parts = partitioning_with_parmetis(pg, agent_num)
        # cut, parts = partitioning_with_metis(pg.setup_graph(False), agent_num, obj)
    elif method == "AdaptiveRepart":
        cut, parts = repartitioning_with_parmetis(pg, agent_num, parts, obj, itr=1000)
    elif method == "FENNEL" or method == "LDG":
        cut, parts = streaming_graph_partition(pg, agent_num, method, iteration)
    elif method == "random":
        parts = np.random.randint(agent_num, size=len(pg.keyframes))
        cut = -1
    elif method == "id":
        parts = []
        for i in range(len(pg.keyframes)):
            num_per_agent = len(pg.keyframes)/agent_num
            agent_id = round(i/num_per_agent)
            if agent_id >= agent_num:
                agent_id = agent_num - 1
            parts.append(agent_id)
        cut = -1
    elif method == "union":
        parts = [0 for i in range(len(pg.keyframes))]
        cut = -1

    index_part = {}
    for i, part in enumerate(parts):
        index_part[i] = part

    if show:
        G = pg.setup_graph(False)
        for i, part in enumerate(parts):
            G._node[i]['color'] = pg.agent_color(part, agent_num, hex=True)
        nx.nx_pydot.write_dot(G, 'pose-graph-partitioning.dot')
        show_graph("pose-graph-partitioning.dot").view()

    _cut, vol, min_keyframes, max_keyframes = pg.repart(agent_num, index_part)
    print(f"New partion obj {cut} agents {agent_num} keyframes {min_keyframes}<->{max_keyframes} inter edges {_cut} comm_vol {vol}")
    return parts
