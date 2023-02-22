from pose_graph_partitioning.utils import *

class KeyFrame():
    def __init__(self, _id, agent_id, pos, quat, is_4d=False, drone_id = -1):
        self.keyframe_id = _id
        self.agent_id = agent_id
        self.drone_id = drone_id
        self.pos = pos
        self.quat = quat
        self.edges = []
        self._connected_keyframe_ids = set()
        self._connected_edges = dict()
        
    def __str__(self):
        #Output Pos Quat WXYZ
        return f"<KeyFrame {self.keyframe_id}\
\tAgent:{self.agent_id} Drone:{self.drone_id}\
\tPos [{self.pos[0]:3.3f} {self.pos[1]:3.3f}  {self.pos[2]:3.3f}]\
\tQuat [{self.quat[0]:3.1f} {self.quat[1]:3.1f} {self.quat[2]:3.1f} {self.quat[3]:3.1f}]\
\t Edges {len(self.edges)}>"
    
    def g2o(self, cvt_id=False,force_ids=None):
        _id = self.keyframe_id
        if cvt_id:
            if force_ids is None:
                _id = convert_keyframe_id(self.agent_id, _id)
            else:
                _id = convert_keyframe_id(force_ids[self.agent_id], _id)

        return f"VERTEX_SE3:QUAT {_id} {self.pos[0]} {self.pos[1]} {self.pos[2]} \
{self.quat[1]} {self.quat[2]} {self.quat[3]} {self.quat[0]}"
    
    def add_edge(self, edge):
        self.edges.append(edge)
        if edge.keyframe_ida != self.keyframe_id:
            self._connected_keyframe_ids.add(edge.keyframe_ida)
            self._connected_edges[edge.keyframe_ida] = edge
        else:
            self._connected_keyframe_ids.add(edge.keyframe_idb)
            self._connected_edges[edge.keyframe_idb] = edge
    
    def get_edge_to_id(self, _id):
        if _id in self._connected_edges:
            return self._connected_edges[_id]
        return None

    def clear_edges(self):
        self.edges = []

    def connected_keyframe_ids(self):
        return self._connected_keyframe_ids

    def copy(self):
        cp = KeyFrame(self.keyframe_id, self.agent_id, self.pos, self.quat, drone_id=self.drone_id)
        for edge in self.edges:
            cp.add_edge(edge.copy())
        return cp

    def csv(self, ts=None):
        if ts is None:
            return f"{self.keyframe_id},{self.pos[0]},{self.pos[1]},{self.pos[2]},{self.quat[0]},{self.quat[1]},{self.quat[2]},{self.quat[3]}"
        else:
            return f"{ts} {self.pos[0]} {self.pos[1]} {self.pos[2]} {self.quat[0]} {self.quat[1]} {self.quat[2]} {self.quat[3]}"
        