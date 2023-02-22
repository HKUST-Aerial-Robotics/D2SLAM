from pose_graph_partitioning.utils import *

class Edge():
    def __init__(self, _ida, _idb, pos, quat, is_inter, is_4d=False, inf_mat=np.eye(6, 6)):
        self.keyframe_ida = _ida
        self.keyframe_idb = _idb
        self.pos = pos
        self.quat = quat
        self.is_inter = is_inter
        self.agent_ida = -1
        self.agent_idb = -1
        self.information_matrix = inf_mat.copy()
    
    def g2o(self, cvt_id=False,force_ids=None):
        _ida = self.keyframe_ida
        _idb = self.keyframe_idb
        if cvt_id:
            if force_ids is None:
                _ida = convert_keyframe_id(self.agent_ida, _ida)
                _idb = convert_keyframe_id(self.agent_idb, _idb)
            else:
                if self.agent_ida in force_ids and self.agent_idb in force_ids:
                    _ida = convert_keyframe_id(force_ids[self.agent_ida], _ida)
                    _idb = convert_keyframe_id(force_ids[self.agent_idb], _idb)
                else:
                    # print(self, "not in ", force_ids)
                    return ""
            
        _str =  f"EDGE_SE3:QUAT {_ida} {_idb} \
{self.pos[0]} {self.pos[1]} {self.pos[2]} \
{self.quat[1]} {self.quat[2]} {self.quat[3]} {self.quat[0]}"
        for i in range(6):
            for j in range(i, 6):
                _str += f" {self.information_matrix[i][j]}"
        return _str

    def __str__(self):
        #Output Pos Quat WXYZ
        _inf_dig = np.diagonal(self.information_matrix)
        return f"<Edge ({self.keyframe_ida}<->{self.keyframe_idb}\
\tAgent {self.agent_ida}<->{self.agent_idb} \
\tPos [{self.pos[0]:3.3f} {self.pos[1]:3.3f}  {self.pos[2]:3.3f}]\
\tQuat [{self.quat[0]:3.1f} {self.quat[1]:3.1f} {self.quat[2]:3.1f} {self.quat[3]:3.1f}\
\tQuat [{_inf_dig[0]:3.1f} {_inf_dig[1]:3.1f} {_inf_dig[2]:3.1f} {_inf_dig[3]:3.1f} \
{_inf_dig[4]:3.1f} {_inf_dig[5]:3.1f}"

    def copy(self):
        return Edge(self.keyframe_ida, self.keyframe_idb, self.pos, self.quat, self.is_inter, 
                inf_mat=self.information_matrix)
