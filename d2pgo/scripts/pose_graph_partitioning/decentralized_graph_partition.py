from pose_graph_partitioning.pose_graph import *
from pose_graph_partitioning.pose_graph_partitioning import *
import time

SIZE_ID_PART = 8 #INT32 + INT32
SIZE_KEYFRAME = 40 
#INT32 KFID| INT32 DRONE_ID| INT32 AGENT_ID| float32*7 40Bytes.
SIZE_POSE_EXCHANGE_ITERATIONS = 32 #INT32 KFID POSE float32*7 32

def convert_to_index_partitions(parts, agents_id_index, default_agent_id=0):
    _new_parts = []
    for part in parts:
        if part in agents_id_index:
            _new_parts.append(part)
        else:
            #Current agent is not available because the change of network topology, we reindex it, give it a default id now.
            _new_parts.append(default_agent_id)
    return np.array(_new_parts)

def convert_from_index_partitions(parts, available_agents, default_agent_id=0):
    #The parts here contains the real id,
    #We need to reindex this id to the idnex in available_agents
    agents_id_index = {}
    for i in range(len(available_agents)):
        agents_id_index[available_agents[i]] = i
    _new_parts = []

    for part in parts:
        _new_parts.append(available_agents[part])
    return _new_parts

class NetworkTopology:
    def __init__(self, agents, direct_connections=None, network_mode="router"):
        #Agents should be list!
        #Commuincation mode:
        #router: all communincation is constant and stable.
        #adhoc: communicate with only neighbors.
        #mesh: communication on mesh network.
        self.network_mode = network_mode
        self.agents = set(agents)
        self.min_agent_id = min_agent_id = np.min(agents)
        network_status = {}
        connected = {}

        #set network status
        master_ids = {}
        for agenta in agents:
            network_status[agenta] = {}
            connected[agenta] = {}
            for agentb in agents:
                network_status[agenta][agentb] = -1
            if network_mode == "router":
                master_ids[agenta] = min_agent_id
            elif network_mode == "mesh":
                master_ids[agenta] = agenta

        if direct_connections is not None and network_mode == "mesh":
            _cost = 1
            for conna, connb in direct_connections:
                network_status[conna][connb] = 1
                network_status[connb][conna] = 1
                connected[connb][conna] = True
                connected[conna][connb] = True
                if master_ids[conna] > master_ids[connb]:
                    master_ids[conna] = master_ids[connb]
                else:
                    master_ids[connb] = master_ids[conna]

            network_updated = True
            while network_updated:
                network_updated = False
                for agenta, agentb in direct_connections:
                    for agentc in agents:
                        #If b connected to c, a-b + b-c < a-c or a-c<0
                        #Then update a-c
                        if agenta == agentc:
                            continue
                        b2c = network_status[agentb][agentc]
                        if b2c > 0:
                            if (network_status[agenta][agentc] < 0 or b2c + _cost < network_status[agenta][agentc]):
                                network_status[agenta][agentc] = b2c + _cost
                                network_status[agentc][agenta] = b2c + _cost
                                connected[agenta][agentc] = True
                                connected[agentc][agenta] = True
                                network_updated = True
                            
                            if master_ids[agenta] > master_ids[agentc]:
                                master_ids[agenta] = master_ids[agentc]
                            else:
                                master_ids[agentc] = master_ids[agenta]

            # print(direct_connections, network_status)

        self._network_status = network_status
        self.master_ids = master_ids
        self.connected = connected

        self.clusters = dict()
        for i in self.master_ids:
            _master = self.master_ids[i]
            if _master not in self.clusters:
                self.clusters[_master] = set()
            self.clusters[_master].add(i)
        # print(f"masters {self.clusters}")
    
    def equal(self, topology_new):
        if self.network_mode == "router":
            return True
        elif self.network_mode == "mesh":
            return self.connected == topology_new.connected
    
    def network_status(self):
        return self._network_status
    
    def communication_cost(self, ida, idb):
        if self.network_mode == "router":
            return 1
        else:
            if ida not in self._network_status:
                return -1
            if idb not in self._network_status[ida]:
                return -1
            return self._network_status[ida][idb]

    def connected_agents(self, id):
        ret = []
        if self.network_mode == "router":
            for j in self.agents:
                if j != id:
                    ret.append(j)
        elif self.network_mode == "mesh" or self.network_mode == "bag":
            for j in self._network_status[id]:
                if j != id and self._network_status[id][j] > 0:
                    ret.append(j)
        return ret

    def is_master(self, agent_id):
        if self.network_mode == "router":
            if agent_id == self.min_agent_id:
                return True
        elif self.master_ids[agent_id] == agent_id:
            return True
        return False
    
    def process_swarm_network_status(self, network_status_it):
        self_id = network_status_it[1]
        status = network_status_it[2]
        for drone_conn in status.network_status:
            _id = drone_conn.drone_id
            self._network_status[self_id][_id] = 1 if drone_conn.active else -1

        for agenta in self.agents:
            self.master_ids[agenta] = agenta

        for i in self.agents:
            for j in self.agents:
                if i!=j and self._network_status[i][j] > 1:
                    self._network_status[j][i] = 1
                    if self.master_ids[i] < self.master_ids[j]:
                        self.master_ids[j] = self.master_ids[i]

        network_updated=True
        while network_updated:
            for i in self.agents:
                _master = self.master_ids[i]
                if _master ==i:
                    continue

        self.clusters = dict()
        for i in self.master_ids:
            _master = self.master_ids[i]
            if _master not in self.clusters:
                self.clusters[_master] = set()
            self.clusters[_master].add(i)

class PoseGraphManager():
    #For single agent
    def __init__(self, self_id, adaptive_repart_duration, agent_ids=None, agent_num=10, target_keyframe_num=4000, target_edge_num=6000):
        self.pg = PoseGraph()
        self.self_id = self_id
        self.is_master = False
        self.network_status = {}
        self.drones = set()
        self.last_adaptive_repart_num = 0
        self.current_keyframe_num = 0
        self.adaptive_repart_duration = adaptive_repart_duration
        self.network_topology_changed = False
        self.network_topology = None
        if agent_ids is not None:
            self.pg.initialize_greedy_partition_by_agents(agent_ids, target_keyframe_num, target_edge_num)
        else:
            self.pg.initialize_greedy_partition(agent_num, target_keyframe_num, target_edge_num)
        self.time_cost_repartitioning = 0
        self.count_repartitioning = 0
        self.total_v = 0

    def update_greedy_partition(self, target_keyframe_num, target_edge_num):
        self.pg.update_greedy_partition(target_keyframe_num, target_edge_num)

    def add_remote_keyframe(self, keyframe, part_id=None):
        self.total_v += SIZE_KEYFRAME
        self.add_keyframe(keyframe, part_id)

    def add_keyframe(self, keyframe, part_id=None):
        #If part_id is None then perform Greedy Partition
        if part_id is not None:
            if part_id not in self.pg.agents:
                # print(f"Agent {self.self_id} skipping {part_id}... wait for repart")
                return -1
        agent_id = self.pg.add_keyframe(keyframe, agent_id=part_id, add_edges_from_keyframe=True)
        self.current_keyframe_num = len(self.pg.keyframes)
        return agent_id
    
    def get_keyframe_copy_ids(self, keys):
        ret = []
        for _id in keys:
            ret.append(self.pg.keyframes[_id].copy())
        return ret
    
    def available_keyframe_ids(self):
        return set(self.pg.keyframes.keys())
    
    def adaptive_repart_routine(self, force_repartition):
        network_changed = self.has_network_topology_changed()
        if self.needAdaptiveRepart(network_changed or force_repartition):
            # print("callAdaptiveRepart")
            self.last_adaptive_repart_num = self.current_keyframe_num
            parts, agent_num = self.callAdaptiveRepart()
            self.update_greedy_partition_routine(self.adaptive_repart_duration)
            return parts, agent_num 
        return None, -1

    def update_greedy_partition_routine(self, dn, edge_rate=3):
        cur_keyframe_num = len(self.pg.keyframes)
        edges_num = len(self.pg.edges)
        if edges_num > 0:
            self.update_greedy_partition(cur_keyframe_num + dn, (cur_keyframe_num + dn)/cur_keyframe_num*edges_num)
        else:
            self.update_greedy_partition(cur_keyframe_num + dn, (cur_keyframe_num + dn)*edge_rate)

    def has_network_topology_changed(self):
        ret = self.network_topology_changed
        self.network_topology_changed = False
        return ret

    def needAdaptiveRepart(self, network_changed):
        if not self.network_topology.is_master(self.self_id):
            return False
        if network_changed:
            return True
        if self.current_keyframe_num - self.last_adaptive_repart_num > self.adaptive_repart_duration:
            return True
        return False

    def callAdaptiveRepart(self, obj="vol", itr=1000):
        start = time.time()
        available_agents = self.get_available_agents()
        #The parts here contains the real id,
        #We need to reindex this id to the idnex in available_agents
        
        agent_num = len(available_agents)
        parts = self.pg.current_partition_array()
        agents_id_index = {}
        for i in range(len(available_agents)):
            agents_id_index[available_agents[i]] = i

        #Reindex the partition to index from 0 to n
        # parts = convert_to_index_partitions(parts, agents_id_index)
        self.pg.update_edges()
        val, parts = repartitioning_with_parmetis(self.pg, agent_num, parts, obj, itr=itr)
        # val, parts = partitioning_with_parmetis(self.pg, agent_num, ubvec=1.1)
        # val, parts = partitioning_with_metis(self.pg.setup_graph(False), agent_num, obj)
        #Restore the partition to available agent id
        parts = convert_from_index_partitions(parts, available_agents)

        self.pg.repart(agent_num, parts, agent_list=available_agents)
        id_parts = {}
        for index in range(len(parts)):
            _id = self.pg.index2id[index]
            id_parts[_id] = parts[index]
        
        self.time_cost_repartitioning += time.time() - start
        self.count_repartitioning+= 1
        return id_parts, agent_num

    def sync_full_partitions(self, agent_num, id_parts, agent_list, addition_kfs):
        self.total_v += len(id_parts)*SIZE_ID_PART #The major overhead is cause by this
        self.pg.repart(agent_num, id_parts=id_parts, agent_list=agent_list, addition_kfs=addition_kfs)
        self.update_greedy_partition_routine(self.adaptive_repart_duration)

    def sync_remote_poses(self, addition_kfs):
        self.total_v+= len(addition_kfs)*SIZE_KEYFRAME
        for keyframe in addition_kfs:
            self.pg.add_keyframe(keyframe, agent_id=keyframe.agent_id, add_edges_from_keyframe=True)

    def get_available_agents(self):
        available_agents = self.network_topology.connected_agents(self.self_id)
        available_agents.append(self.self_id)
        available_agents.sort()
        return available_agents
    
    def set_network_topology(self, network_topology):
        #Network status is only 0 or 1 now.
        #In the further versions we may add additional network quality informations.
        if self.network_topology is not None:
            self.network_topology_changed = not self.network_topology.equal(network_topology)
        self.network_topology = network_topology
        self.network_status = network_topology.network_status()
    
    def get_poses_need_to_solve(self, _id):
        return self.pg.agents[_id].get_keyframe_ids()

    def get_self_poses_need_to_solve(self):
        return self.get_self_poses_need_to_solve(self.self_id)

    def statistical(self):
        cut, vol, min_keyframes, max_keyframes, kf_num, edge_num = self.pg.statistical()
        return cut, vol, min_keyframes, max_keyframes, kf_num, edge_num 

    def edge_num(self):
        return len(self.pg.edges)

    def keyframe_num(self):
        return len(self.pg.keyframes)

    def show(self, title, ax=None):
        return self.pg.show(title, ax=ax)