from math import *

FENNEL_PARTITIONING = "FENNEL"
LDG_PARTITIONING = "LDG"

class GreedyPartitioningParam:
    def __init__(self, k, n, m, gamma=1.5, nu=1.1):
        self.k = k #k is the number of the agents
        self.n = n #n is the number of total vertexs
        self.m = m #m is the number of total edges

        self.gamma = gamma
        self.nu = nu
        # print(f"Number of agents {k} total vertexs {n} edges {m} maxsize {self.FENNEL_maxpartsize()}")

    def alpha(self):
        return self.m * (self.k**(self.gamma-1))/(self.n**self.gamma)

    def C(self):
        _C = self.n / self.k
        return _C
    
    def FENNEL_maxpartsize(self):
        return self.nu*self.n/self.k
    
class GreedyPartitioning:
    def __init__(self, param, method=FENNEL_PARTITIONING):
        self.method = method
        self.partition_set = {}
        self.partition_of_id = {}
        self.param = param

        for k in range(param.k):
            self.partition_set[k] = set()

    def update_greedy_partition(self, agent_num, partition_set, partition_of_id, target_keyframe_num, target_edge_num):
        self.param.k = agent_num
        self.param.n = target_keyframe_num
        self.param.m = target_edge_num
        self.partition_set = partition_set
        self.partition_of_id = partition_of_id
        # print(f"agent_num {agent_num}, target_keyframe_num {target_keyframe_num}, target_edge_num {target_edge_num} max size {self.param.FENNEL_maxpartsize()}")

    def partition_vertex(self, vertex, adjacency_list, old_partitioning=None):
        h_max = -299792458
        best_part_id = -1
        for part_id in self.partition_set:
            if self.method == FENNEL_PARTITIONING:
                if len(self.partition_set[part_id]) >= self.param.FENNEL_maxpartsize():
                    # print(f"Part {part_id} size {len(self.partition_set[part_id]) } maxsize {self.param.FENNEL_maxpartsize()}")
                    continue

            h = self.heuristic(adjacency_list, part_id, old_partitioning)
            if h > h_max:
                best_part_id = part_id
                h_max = h
        
        if best_part_id >= 0:
            self.assign_vertex_with_partition(vertex, best_part_id)
        else:
            print(f"vertex {vertex} heuristic {h_max} assigned to -1, bugs occurred partition_set {len(self.partition_set)}")
        # print(f"vertex {vertex} heuristic {h_max} assigned to {best_part_id} size {len(self.partition_set[part_id])}")
        return best_part_id
    
    def assign_vertex_with_partition(self, vertex, part_id):
        self.partition_set[part_id].add(vertex)
        self.partition_of_id[vertex] = part_id

    def heuristic(self, adjacency_list, part_id, old_partitioning=None):
        if self.method == FENNEL_PARTITIONING:
            return self.heuristic_FENNEL(adjacency_list, part_id, old_partitioning)
        elif self.method == LDG_PARTITIONING:
            return self.heuristic_LDG(adjacency_list, part_id, old_partitioning)
    
    def heuristic_LDG(self, adjacency_list, part_id, old_partitioning=None):
        cur_par_num = len(self.partition_set[part_id])
        neighbor_num = self.neighbor_vertex_num(adjacency_list, part_id)
        C = self.param.C()
        w = 1 - cur_par_num/C
        # print(f"part_id {part_id} cur_par_num {cur_par_num} C {C} w {w} res {neighbor_num*w}")
        return neighbor_num*w

    def heuristic_FENNEL(self, adjacency_list, part_id, old_partitioning=None):
        cur_par_num = len(self.partition_set[part_id])
        if old_partitioning is not None:
            neighbor_num = old_partitioning.neighbor_vertex_num(adjacency_list, part_id)
        else:
            neighbor_num = self.neighbor_vertex_num(adjacency_list, part_id)

        alpha = self.param.alpha()
        gamma = self.param.gamma
        # print(f"part_id {part_id} neighbor_num {neighbor_num} cur_par_num {cur_par_num} alpha{alpha} gamma{gamma} dc {alpha*gamma*sqrt(cur_par_num)} adjacency_list {adjacency_list}")
        return  neighbor_num - alpha*gamma*(cur_par_num**(gamma-1))
    
    def neighbor_vertex_num(self, adjacency_list, part_id):
        count = 0
        for vertex in adjacency_list:
            if vertex in self.partition_of_id and self.partition_of_id[vertex] == part_id:
                count += 1
        return count

if __name__ == "__main__":
    pass

