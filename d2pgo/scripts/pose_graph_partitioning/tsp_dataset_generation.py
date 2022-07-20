from pose_graph_partitioning.pose_graph import *
def cvt_tsp(pg, start_points=None, agent_num=-1, random_weight = True):
    if len(pg.id2index) == 0:
        print("You must run setup_graph first")
        return

    force_starts = start_points is not None
    min_size = 3
    max_size = len(pg.keyframes)*2
    if force_starts:
        agent_num = len(start_points)
        sp_set = set()
        for _id in start_points:
            sp_set.add(pg.id2index[_id])
        print("start index", sp_set)
        
    keyframe_num = len(pg.keyframes)

    max_weight = 1000000
    min_weight = -  1000
    with open("mtsp.par", "w") as f:
        print(f"""SPECIAL
PROBLEM_FILE=problem.tsp
SALESMEN = {agent_num}
MTSP_OBJECTIVE = MINSUM
MTSP_MIN_SIZE = {min_size}
MTSP_MAX_SIZE = {max_size}
RUNS = 5
TOUR_FILE = mtsp.tour
MTSP_SOLUTION_FILE=mtsp.sol
TRACE_LEVEL =0""",file=f)

    weight_mat = ""
    # We will set a virtual start point on index agent id
    # The weight from virtual start point to some start point is -100000
    dim = keyframe_num+1
    weight = np.ones((dim, dim), dtype=np.int)*max_weight
    for _i in range(dim):
        for _j in range(dim):
            i = _i - 1
            j = _j - 1
            if i == -1:
                if force_starts:
                    if j in sp_set:
                        weight[_i, _j] = min_weight
                else:
                    weight[_i, _j] = min_weight
                    
            if j == -1:
                    weight[_i, _j] = min_weight
            elif i >=0 and j >=0 and pg.has_edge(pg.index2id[i], pg.index2id[j]):
                if random_weight:
                    weight[_i, _j] = random.randint(-10, 10)
                else:
                    weight[_i, _j] = 0
            

    for i in range(dim):
        for j in range(dim):
            weight_mat += f"{weight[i, j]}\t"
        weight_mat += "\n"

    with open("problem.tsp", "w") as f:
        print(f"""NAME : mtsp
TYPE: ATSP
DIMENSION: {dim}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
EDGE_WEIGHT_SECTION
{weight_mat}""", file=f)

def solve_tsp():
    s = os.popen('LKH mtsp.par')
    output = s.read()
    print("LKH:")
    pathes = []
    with open("mtsp.sol", "r") as f:
        lines = f.readlines()
        for i in range(2, len(lines)):
            line = lines[i]
            path = []
            c = 0
            for item in line.split(' '):
                if item == "1":
                    c += 1
                if c < 2:
                    path.append(int(item)-2)
                else:
                    break
            pathes.append(path[1:])
    print(f"Read {len(pathes)} pathes from TSP solution")
    return pathes

def make_edge(kfa, kfb):
    pos = kfb.pos - kfa.pos
    pos = quaternion_rotate(quaternion_inverse(kfa.quat), pos)
    quat = quaternion_multiply(quaternion_inverse(kfa.quat), kfb.quat)
    quat = unit_vector(quat)
    edge = Edge(kfa.keyframe_id, kfb.keyframe_id, pos, quat, False, False)
    return edge

def fix_path_disconnected(pg, pathes, pg_optimized, align_beginning=False):
    count_disconnected = 0
    for path in pathes:
        for i in range(0, len(path)-1):
            _ida = pg.index2id[path[i]]
            _idb = pg.index2id[path[i+1]]
            if not pg.has_edge(_ida, _idb):
                count_disconnected += 1
                kfa = pg_optimized.keyframes[_ida]
                kfb = pg_optimized.keyframes[_idb]
                edge = make_edge(kfa, kfb)
                edge.information_matrix = pg.edges[0].information_matrix
                pg.edges.append(edge)
    
    path_num = len(pathes)
    if align_beginning:
        for i in range(path_num-1):
            _ida = pg.index2id[pathes[i][0]]
            _idb = pg.index2id[pathes[i+1][0]]
            if not pg.has_edge(_ida, _idb):
                count_disconnected += 1
                kfa = pg_optimized.keyframes[_ida]
                kfb = pg_optimized.keyframes[_idb]
                edge = make_edge(kfa, kfb)
                edge.information_matrix = pg.edges[0].information_matrix
                pg.edges.append(edge)

    pg.update_edges()
    print(f"Fix {count_disconnected} disconnected edges now edges {len(pg.edges)}")

def generate_path_tsp(pg, agent_num, random_weight = True, pg_optimized = None, fix_path=True, align_beginning=True, pathes=None):
    if pathes is None:
        cvt_tsp(pg, agent_num = agent_num, random_weight = random_weight)
        pathes = solve_tsp()
    if fix_path:
        fix_path_disconnected(pg, pathes, pg_optimized, align_beginning = align_beginning)
    index_part = {}
    for i in range(len(pathes)):
        for index in pathes[i]:
            index_part[index] = i
    cut, vol, min_keyframes, max_keyframes = pg.repart(agent_num, index_part)
    for _index in index_part:
        pg.keyframes[pg.index2id[_index]].drone_id = index_part[_index]
    print(f"TSP new path keyframes {min_keyframes}<->{max_keyframes} cut {cut} comm_vol {vol}")
    return pathes
