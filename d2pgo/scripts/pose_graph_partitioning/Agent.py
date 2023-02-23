
import numpy as np
class Agent():
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.keyframes = []
        self.kfs = {}
        self.edges = []
        self.inter_edge_num = 0
        
    def add_keyframe(self, kf):
        kf.clear_edges()
        self.keyframes.append(kf)
        self.kfs[kf.keyframe_id] = kf
    
    def has_keyframe(self, kf_id):
        return kf_id in self.kfs
    
    def get_keyframe_ids(self):
        return list(self.kfs.keys())
    
    def add_edge(self, edge):
        if edge.keyframe_ida in self.kfs:
            self.kfs[edge.keyframe_ida].add_edge(edge)
        if edge.keyframe_idb in self.kfs:
            self.kfs[edge.keyframe_idb].add_edge(edge)
        self.edges.append(edge)
        if edge.is_inter:
            self.inter_edge_num += 1

    def clear_edges(self):
        self.edges = []
        for kfid in self.kfs:
            self.kfs[kfid].clear_edges()
    
    def write_to_g2o(self, path, cvt_id=False, addition_edges = [], force_ids=None, add=False):
        mode = "w"
        if add:
            mode = "a"
        with open(path, mode) as f:
            for keyframe in self.keyframes:
                print(keyframe.g2o(cvt_id,force_ids), file=f)
            
            for edge in self.edges:
                print(edge.g2o(cvt_id,force_ids), file=f)

            for edge in addition_edges:
                print(edge.g2o(cvt_id,force_ids), file=f)
    
    def write_to_csv(self, path, idx_stamp_map = None):
        with open(path, 'w') as f:
            for keyframe in self.keyframes:
                if idx_stamp_map is not None:
                    stamp = idx_stamp_map[keyframe.keyframe_id]
                else:
                    stamp=None
                print(keyframe.csv(stamp), file=f)

    def check_connected_keyframes(self, kf_id):
        visited_keyframes = set()
        visited_keyframes.add(self.kfs[kf_id].keyframe_id)
        keyframe_queue = [self.kfs[kf_id]]
        while len(keyframe_queue) > 0:
            _keyframe = keyframe_queue.pop(0)
            for edge in _keyframe.edges:
                if edge.keyframe_ida == _keyframe.keyframe_id:
                    _idb = edge.keyframe_idb
                else:
                    _idb = edge.keyframe_ida
                if _idb not in visited_keyframes and _idb in self.kfs :
                    visited_keyframes.add(_idb)
                    keyframe_queue.append(self.kfs[_idb])
        return visited_keyframes

    def check_agent_connection(self):
        clusters = []
        if len(self.keyframes) == 0:
            return False, clusters
        visited_keyframes = set()
        clusters.append(self.check_connected_keyframes(self.keyframes[0].keyframe_id))
        visited_keyframes = visited_keyframes | clusters[0]
        num_visited_keyframes = len(clusters[0])
        if num_visited_keyframes == len(self.keyframes):
            return True, clusters
        
        ptr = 1
        while num_visited_keyframes < len(self.keyframes) and ptr < len(self.keyframes):
            if self.keyframes[ptr].keyframe_id not in visited_keyframes:
                new_cluster = self.check_connected_keyframes(self.keyframes[ptr].keyframe_id)
                visited_keyframes = visited_keyframes | new_cluster
                num_visited_keyframes = num_visited_keyframes + len(new_cluster)
                clusters.append(new_cluster)
            ptr += 1
        return False, clusters


    def evaluate_trajectory_length(self):
        length = 0
        pos_old = None
        for kf in self.keyframes:
            pos = kf.pos
            if pos_old is not None:
                length += np.linalg.norm(pos - pos_old)
            pos_old = pos
        return length