#!/usr/bin/env python3
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from matplotlib import cm
import networkx as nx
import random
from transformations import *
from pose_graph_partitioning.graph_partitioning import *
from datetime import datetime
from pose_graph_partitioning.utils import *
import copy
from typing import List, Set, Dict, Tuple, Optional

random.seed(datetime.now())
from pose_graph_partitioning.Keyframe import KeyFrame
from pose_graph_partitioning.Edge import Edge
from pose_graph_partitioning.Agent import Agent

class PoseGraph():
    edges: List[Edge]
    keyframes: Dict[int, KeyFrame]
    def __init__(self, path = "", single=False):
        self.keyframes = {}
        self.agents = {}
        self.edges = []
        self.inter_edge_num = 0

        self.edge_map = {}
        self.edge_connections = {}
        self.default_greedy_method = FENNEL_PARTITIONING
        self.index2id = {}
        self.id2index = {}
        self.greedy_partitioning = None
        self.partition_set = {}
        self.partition_of_id = {}
        self.agent_ids = []
        if path != "":
            if single:
                self.read_g2o_single(path)
            else:
                self.read_g2o_folder(path, prt=False)

    def agent_num(self):
        return len(self.agents)

    def initialize_greedy_partition(self, agent_num, target_keyframe_num, target_edge_num):
        param = GreedyPartitioningParam(agent_num, target_keyframe_num, target_edge_num, gamma=1.5)
        self.greedy_partitioning = GreedyPartitioning(param=param, method="FENNEL")
        for i in range(agent_num):
            self.agents[i] = Agent(i)
            self.partition_set[i] = set()
            self.agent_ids.append(i)
    
    def initialize_greedy_partition_by_agents(self, agent_ids, target_keyframe_num, target_edge_num):
        param = GreedyPartitioningParam(len(agent_ids), target_keyframe_num, target_edge_num, gamma=1.5)
        self.greedy_partitioning = GreedyPartitioning(param=param, method="FENNEL")
        self.agent_ids = agent_ids
        for i in agent_ids:
            self.agents[i] = Agent(i)
            self.partition_set[i] = set()
    
    def update_greedy_partition(self, target_keyframe_num, target_edge_num):
        self.greedy_partitioning.update_greedy_partition(self.agent_num(), self.partition_set, self.partition_of_id, 
                target_keyframe_num, target_edge_num)

    def add_keyframe(self, kf, agent_id = None, add_edges_from_keyframe = False):
        _index = len(self.keyframes)
        _id = kf.keyframe_id
        self.keyframes[_id] = kf
        self.id2index[_id] = _index
        self.index2id[_index] = _id
        edges = kf.edges
        kf.edges = []
        if agent_id is None:
            #Here we will 
            adjacency_list = []
            for edge in edges:
                if edge.keyframe_ida == _id:
                    adjacency_list.append(edge.keyframe_idb)
                else:
                    adjacency_list.append(edge.keyframe_ida)
            agent_id = self.greedy_partitioning.partition_vertex(_id, adjacency_list) #Note this id is start from 0!
        
        if agent_id not in self.agents:
            self.agents[agent_id] = Agent(agent_id)
            # print(f"graph contains no {agent_id} created")
        kf.agent_id = agent_id
        self.agents[agent_id].add_keyframe(kf)

        if agent_id not in self.partition_set:
            self.partition_set[agent_id] = set()

        self.partition_set[agent_id].add(kf.keyframe_id)
        self.partition_of_id[kf.keyframe_id] = agent_id

        if add_edges_from_keyframe:
            for edge in edges:
                succ = self.add_edge(edge)
                if succ:
                    self.agents[agent_id].add_edge(edge)
        return agent_id

    def add_edge(self, edge):
        if edge.keyframe_ida not in self.keyframes or edge.keyframe_idb not in self.keyframes:
            #edge not in keyframes
            return False

        edge.agent_id_a = self.keyframes[edge.keyframe_ida].agent_id
        edge.agent_id_b = self.keyframes[edge.keyframe_idb].agent_id
        # print(f"Adding edge {edge}")
        s1 = f"{edge.keyframe_ida}<->{edge.keyframe_idb}"
        s2 = f"{edge.keyframe_idb}<->{edge.keyframe_ida}"

        if s1 in self.edge_map or s2 in self.edge_map:
            return False
        else:
            self.edge_map[s1] = edge
            self.edge_map[s2] = edge
            self.edges.append(edge)
        
        if edge.keyframe_ida in self.keyframes:
            self.keyframes[edge.keyframe_ida].add_edge(edge)
        if edge.keyframe_idb in self.keyframes:
            self.keyframes[edge.keyframe_idb].add_edge(edge)

        if edge.keyframe_ida not in self.edge_connections:
            self.edge_connections[edge.keyframe_ida] = set()

        if edge.keyframe_idb not in self.edge_connections:
            self.edge_connections[edge.keyframe_idb] = set()

        self.edge_connections[edge.keyframe_ida].add(edge.keyframe_idb)
        self.edge_connections[edge.keyframe_idb].add(edge.keyframe_ida)
        return True

    def has_edge(self, ida, idb):
        if ida in self.edge_connections:
            return idb in self.edge_connections[ida]
        return False

    def update_edges(self, duplicate_inter_edge=True):
        for agent_id in self.agents:
            self.agents[agent_id].clear_edges()
        for keyframe_id in self.keyframes:
            self.keyframes[keyframe_id].clear_edges()
            
        cut = 0
        count = 0
        for edge in self.edges:
            ida = edge.keyframe_ida
            idb = edge.keyframe_idb
            try:
                agent_id_a = self.keyframes[ida].agent_id
                agent_id_b = self.keyframes[idb].agent_id
                edge.agent_ida = agent_id_a
                edge.agent_idb = agent_id_b
                # print(edge)
                idb= edge.keyframe_idb
                if agent_id_a != agent_id_b:
                    cut += 1
                    edge.is_inter = True
                    # print(f"Edge {edge}")
                else:
                    edge.is_inter = False
                self.agents[agent_id_a].add_edge(edge)
                count += 1
                if agent_id_b != agent_id_a and duplicate_inter_edge:
                    self.agents[agent_id_b].add_edge(edge)
            except:
                raise Exception(f"Unknown agent! {edge}")

        # print(f"Update {count} edge")        
        return len(self.edges), cut
    
    def agent_color(self, agent_id, agent_num, hex=False):
        c = cm.get_cmap("jet")(agent_id/ agent_num)
        if hex:
            return '#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255))
        return c    
    
    def cut(self):
        cut = 0
        for edge in self.edges:
            ida = edge.keyframe_ida
            idb = edge.keyframe_idb
            try:
                agent_id_a = self.keyframes[ida].agent_id
                agent_id_b = self.keyframes[idb].agent_id
                # print(edge)
                idb = edge.keyframe_idb
                if agent_id_a != agent_id_b:
                    cut += 1
                    edge.is_inter = True
                else:
                    edge.is_inter = False
            except:
                raise Exception(f"Unknown agent! {edge}")
        return cut

    def statistical(self):
        max_keyframes = 0
        min_keyframes = 10000
        
        for agent_id in self.agents:
            keyframe_num = len(self.agents[agent_id].keyframes)
            if keyframe_num > max_keyframes:
                max_keyframes = keyframe_num
            if keyframe_num < min_keyframes:
                min_keyframes = keyframe_num

        return self.cut(), self.communication_volume(), min_keyframes, max_keyframes, len(self.keyframes), len(self.edges)

    def communication_volume(self, broadcast=False):
        _v = 0
        agent_vol = {}
        edge_keyframes = {}
        for i in self.agents:
            edge_keyframes[i] = set()
        for kf_id in self.keyframes:
            kf = self.keyframes[kf_id]
            if kf.agent_id not in agent_vol:
                agent_vol[kf.agent_id] = 0
            _agent_set = set()
            for edge in kf.edges:
                _agent_ida = self.keyframes[edge.keyframe_ida].agent_id
                _agent_idb = self.keyframes[edge.keyframe_idb].agent_id
                if _agent_ida != kf.agent_id:
                    _agent_set.add(_agent_ida)
                    edge_keyframes[kf.agent_id].add(kf.keyframe_id)
                elif _agent_idb != kf.agent_id:
                    _agent_set.add(_agent_idb)
                    edge_keyframes[kf.agent_id].add(kf.keyframe_id)
            _v += len(_agent_set)
            agent_vol[kf.agent_id] += len(_agent_set)
        if broadcast:
            vol = 0
            for i in self.agents:
                vol += len(edge_keyframes[i])
            return vol
        else:
            return _v

    def setup_graph(self, show=False):
        G = nx.Graph()
        self.G =G
        # Set up graph structure
        _edges = []
        
        id2index = self.id2index
        index2id = self.index2id
        for edge in self.edges:
            try:
                index_a = id2index[edge.keyframe_ida]
                index_b = id2index[edge.keyframe_idb]
                _edges.append((index_a, index_b))
            except:
                pass

        G.add_nodes_from(range(len(index2id)))
        G.add_edges_from(_edges)
        nx.nx_pydot.write_dot(G, 'pose-graph.dot')
        for i in index2id:
            _id = index2id[i]
            G._node[i]['color'] = self.agent_color(self.keyframes[_id].agent_id, self.agent_num(), hex=True)
        nx.nx_pydot.write_dot(G, 'pose-graph.dot')
        if show:
            show_graph("pose-graph.dot").view()
        return G

    def show_graphviz(self, title="posegraph"):
        dot = graphviz.Graph(comment=title, graph_attr={
            "label":"Pose Graphs",
		    "shape":"ellipse",
		    "style":"filled"
        })
        # for _id in self.keyframes:
            # dot.node(f'{_id}', f'Node {_id}')
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            c = graphviz.Graph(name=f"cluster_{agent_id}", 
                graph_attr={'shape': 'ellipse', "label":f"solver{agent_id}"})
            for _id in agent.kfs:
                c.node(f'{_id}', f'Node {_id}@Drone{agent.kfs[_id].drone_id}')
            dot.subgraph(c)

        for edge in self.edges:
            _ida = edge.keyframe_ida
            _idb = edge.keyframe_idb
            if edge.is_inter:
                dot.edge(f'{_ida}', f'{_idb}', constraint='true', dir="none", color='blue')
            else:
                dot.edge(f'{_ida}', f'{_idb}', constraint='true', dir="none", color='black')

        dot.render(f"{title}-graphviz.gv", view=True)  

    def repart(self, agent_num, index_part=None, id_parts=None, agent_list=None, addition_kfs=[]):
        self.agents.clear()
        self.partition_set.clear()
        self.partition_of_id.clear()

        for kf in addition_kfs:
            _index = len(self.keyframes)
            self.keyframes[kf.keyframe_id] = kf
            self.id2index[kf.keyframe_id] = _index
            self.index2id[_index] = kf.keyframe_id
            edges = copy.copy(kf.edges)
            for edge in edges:
                self.add_edge(edge)

        if id_parts is None:
            id_parts = {}
            for index in self.index2id:
                _id = self.index2id[index]
                id_parts[_id] = index_part[index]

        if agent_list is None:
            agent_list = range(agent_num)

        for i in agent_list:
            self.agents[i] = Agent(i)

        for _id in id_parts:
            _agent_id = id_parts[_id]
            self.keyframes[_id].agent_id = _agent_id
            self.agents[_agent_id].add_keyframe(self.keyframes[_id])

            if _agent_id not in self.partition_set:
                self.partition_set[_agent_id] = set()
            self.partition_set[_agent_id].add(_id)
            self.partition_of_id[_id] = _agent_id

        if self.greedy_partitioning is not None:
            self.greedy_partitioning.partition_set = self.partition_set
            self.greedy_partitioning.partition_of_id = self.partition_of_id

        _, inter_edges = self.update_edges()
        cut, vol, min_keyframes, max_keyframes, _, _ = self.statistical()
        return cut, vol,  min_keyframes, max_keyframes

    def search_shorest_pathes_of_clusters(self, clusters):
        #Pathes from clusters[0] to clusters[1:-1]
        S_ = {i:0 for i in clusters[0]}
        Q_ = {i:10000000 for i in self.keyframes}
        update_vertices_set = set()
        Paths_ = {}
        required_set = {} #keyframe_id:cluster_ud
        required_set_remain = set()
        ret = []
        for i in range(1, len(clusters)):
            for _id in clusters[i]:
                required_set[_id] = i
            required_set_remain.add(i)

        #Initialization the distance of Q
        for v in S_:
            del Q_[v]
        for _id in S_:
            Paths_[_id] = [_id]
            connected_ids = self.keyframes[_id].connected_keyframe_ids()
            for _idq in connected_ids:
                if _idq in Q_:
                    Q_[_idq] = 1
                    update_vertices_set.add(_idq)
                    Paths_[_idq] = [_id, _idq]

        #Start iteration
        while len(required_set_remain) > 0:
            #First get the vertex with min 
            min_dist = 10000
            min_dist_id = -1
            for _id in update_vertices_set:
                if Q_[_id] < min_dist:
                    min_dist = Q_[_id]
                    min_dist_id = _id

            # print(f"min_dist_id {min_dist_id} update_vertices_set {update_vertices_set} required_set_remain {required_set_remain} required_set {required_set}")

            if min_dist_id < 0:
                break

            #Update accroding to min_dist_id
            S_[min_dist_id]  = Q_[min_dist_id]
            del Q_[min_dist_id]
            update_vertices_set.remove(min_dist_id)
            if min_dist_id in required_set:
                #Remove the cluster set.
                if required_set[min_dist_id] in required_set_remain:
                    #This is shortest path to cluster!
                    ret.append(Paths_[min_dist_id])
                    required_set_remain.remove(required_set[min_dist_id])

            #Update vertices connect to min_dist_id
            connected_ids = self.keyframes[min_dist_id].connected_keyframe_ids()
            for _idq in connected_ids:
                if _idq in Q_ and Q_[_idq] > S_[min_dist_id] + 1:
                    Q_[_idq] = S_[min_dist_id] + 1
                    update_vertices_set.add(_idq)
                    Paths_[_idq] = Paths_[min_dist_id].copy()
                    Paths_[_idq].append(_idq)
        #Finish Dijkstra

        #Return pathes from clusters[0] to clusters[1] .. clusters[n]
        return ret

    def construct_edge_from_path(self, path):
        edge = None
        cov = np.zeros((6, 6))
        for i in range(len(path)-1):
            _ida = path[i]
            kfa = self.keyframes[_ida]
            _idb = path[i+1]
            kfb = self.keyframes[_idb]
            _edge = kfa.get_edge_to_id(_idb)

            if _edge.keyframe_ida != _ida:
                _quat = quaternion_inverse(_edge.quat)
                _pos = quaternion_rotate(_quat, -_edge.pos)
                _edge = Edge(_ida, _idb, _pos, _quat, False, inf_mat=_edge.information_matrix)
            else:
                _pos = _edge.pos
                _quat = _edge.quat

            if edge is None:
                edge = _edge.copy()
                edge.keyframe_ida = _ida
                edge.keyframe_idb = _idb
                edge.agent_ida = kfa.agent_id
                edge.agent_idb = kfb.agent_id
                cov += np.linalg.inv(edge.information_matrix)
            else:
                _pos = quaternion_rotate(edge.quat, _pos) + edge.pos
                _quat = quaternion_multiply(edge.quat, _quat)
                edge.pos = _pos
                edge.quat = _quat
                edge.keyframe_idb = _idb
                edge.agent_idb = kfb.agent_id
                cov += np.linalg.inv(edge.information_matrix)
        edge.information_matrix = np.linalg.inv(cov)
        # print(f"path {path}constructed edge", edge.information_matrix)
        return edge
        
    def ComplementaryAgentInternalConnections(self, agent_id):
        connected, clusters = self.agents[agent_id].check_agent_connection()
        # print(f"agent_id {agent_id} cluster {clusters}")
        if connected or len(clusters) == 0:
            return []
        # print(f"Agent {agent_id} is not internal connected, clusters {len(clusters)}...")
        pathes = self.search_shorest_pathes_of_clusters(clusters)
        # print(f"Searched pathes: {pathes}")

        #Now we construct the edges from these pathes
        edges = [self.construct_edge_from_path(path) for path in pathes]
        return edges

        
    def cvt_CSR(self, debug=False):
        kf_num = len(self.keyframes)
        keyframes_index = np.linspace(0, kf_num-1, kf_num, dtype=np.int)
        adjncy = []
        xadj = []

        connected_graph = [set() for _ in range(kf_num)]

        count_edges = 0
        for _index in keyframes_index:
            _id = self.index2id[_index]
            kf = self.keyframes[_id]

            count_edges += len(kf.edges)
            for edge in kf.edges:
                if edge.keyframe_ida == _id:
                    if edge.keyframe_idb in self.id2index and edge.keyframe_ida == _id :
                        if self.id2index[edge.keyframe_idb] == _index:
                            print(f"Error! Edge connected to itself: {edge} keyframe {kf} index {_index}")
                            print(self.id2index[edge.keyframe_ida], self.id2index[edge.keyframe_idb])
                        else:
                            try:
                                index_v = self.id2index[edge.keyframe_idb]
                                connected_graph[_index].add(index_v)
                                connected_graph[index_v].add(_index)
                            except:
                                print(f"index_v {index_v} kf_num {kf_num}")
                elif edge.keyframe_idb == _id  and edge.keyframe_ida in self.id2index:
                    if self.id2index[edge.keyframe_ida] == _index:
                        print(f"Error! Edge connected to itself: {edge} keyframe {kf} index {_index}")
                    else:
                        index_v = self.id2index[edge.keyframe_ida]
                        connected_graph[_index].add(index_v)
                        connected_graph[index_v].add(_index)

        for _index in keyframes_index:
            xadj.append(len(adjncy))
            for k in connected_graph[_index]:
                adjncy.append(k)

        xadj.append(len(adjncy))
        return xadj, adjncy

    def current_partition_array(self):
        res = np.zeros(len(self.keyframes), dtype=np.int)
        for index in range(len(self.keyframes)):
            _id = self.index2id[index]
            res[index] = self.keyframes[_id].agent_id
        return res
        
    def serialize(self, index = 0):
        poses = []
        edges_a = []
        edges_b = []
        kf_mc = []
        line_color = []
        edge_real = []
        inter_edge_color = (0, 0.0, 1.0, 1.0)
        intra_edge_color = (0.4, 0.4, 0.4, 0.5)
        axes_x = []
        axes_y = []
        axes_z = []
        
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            _poses = []
            _axes_x = []
            _axes_y = []
            _axes_z = []
            c = self.agent_color(agent_id+index, self.agent_num()+index)
            m = "."#marker_list[agent_id % marker_num]
            kf_mc.append((m, c))
            for kf in agent.keyframes:
                _poses.append(kf.pos)
                x_vec = quaternion_rotate(kf.quat, np.array([1, 0, 0]))
                y_vec = quaternion_rotate(kf.quat, np.array([0, 1, 0]))
                z_vec = quaternion_rotate(kf.quat, np.array([0, 0, 1]))
                _axes_x.append(x_vec)
                _axes_y.append(y_vec)
                _axes_z.append(z_vec)
            if len(_poses) > 0:
                poses.append(_poses)
                axes_x.append(_axes_x)
                axes_y.append(_axes_y)
                axes_z.append(_axes_z)
        for edge in self.edges:
            pos = self.keyframes[edge.keyframe_ida].pos
            edges_a.append(pos)
            edge_real.append(edge.pos)
            if edge.is_inter:
                line_color.append(inter_edge_color)
            else:
                line_color.append(intra_edge_color)

            posb = self.keyframes[edge.keyframe_idb].pos
            edges_b.append(posb)
            # if np.linalg.norm(posb-pos) > 5:
            #     print(f"Big dpose {edge.keyframe_ida}<->{edge.keyframe_idb}")
            
        return poses, np.array(axes_x),np.array(axes_y),np.array(axes_z), kf_mc, np.array(edges_a), np.array(edges_b)-np.array(edges_a), np.array(edge_real), line_color

    def show(self, title="Posegraph", ax=None, show_raw_edges=False, clear=True, index=0, close=True, 
            axis_len = -1, plot3d=True, show_edges=True, show_title=True, elev=45, azim=45, show_axis_labels=True,
            color=None, marker=None):
        poses, axes_x, axes_y, axes_z, kf_mc, edges_a, edges_b, edge_real, edge_color= self.serialize(index)
        if len(poses) > 1:
            _poses = np.concatenate(poses, axis=0)
        else:
            _poses = np.array(poses[0])
        range_x = np.max(_poses[:,0]) - np.min(_poses[:,0])
        range_y = np.max(_poses[:,1]) - np.min(_poses[:,1])
        range_z = np.max(_poses[:,2]) - np.min(_poses[:,2])
        
        if ax is None:
            if close:
                plt.close(title)
            fig = plt.figure(title)
            plt.tight_layout()
            plt.clf()
            if show_title:
                plt.title(title)
            if plot3d:
                ax = fig.add_subplot(111, projection='3d')
                ax.set_box_aspect((1, range_y/range_x, range_z/range_x))
                if show_axis_labels:
                    ax.set_xlabel("X(m)")
                    ax.set_ylabel("Y(m)")
                    ax.set_zlabel("Z(m)")
                else:
                    ax.set_xlabel("X(m)")
                    ax.set_ylabel("Y(m)")
                    ax.set_axis_off()
                if show_title:
                    ax.set_title(title)
                ax.view_init(elev=elev, azim=azim)

            else:
                ax = fig.add_subplot(111)
                ax.set_box_aspect(range_y/range_x)
                if show_axis_labels:
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
        else:
            if clear:
                ax.clear()
            if plot3d:
                ax.set_box_aspect((1, range_y/range_x, range_z/range_x))
            else:
                ax.set_box_aspect(range_y/range_x)

        for i in range(len(poses)):
            _poses = np.array(poses[i])
            if _poses.shape[0] == 0:
                continue
            _axes_x = axes_x[i]
            _axes_y = axes_y[i]
            _axes_z = axes_z[i]
            m, c = kf_mc[i]
            if not clear:
                c = None
            color = color if color is not None else c
            marker = marker if marker is not None else m
            if plot3d:
                ax.scatter(_poses[:,0], _poses[:,1], _poses[:,2], color=color,marker=marker,label=f"{title} agent {i}")
                if axis_len > 0:
                    ax.quiver(_poses[:,0], _poses[:,1], _poses[:,2], _axes_x[:, 0]*axis_len, _axes_x[:, 1]*axis_len, _axes_x[:, 2]*axis_len, linewidths=-.5, arrow_length_ratio=0,  color="red")
                    ax.quiver(_poses[:,0], _poses[:,1], _poses[:,2], _axes_y[:, 0]*axis_len, _axes_y[:, 1]*axis_len, _axes_y[:, 2]*axis_len, linewidths=-.5, arrow_length_ratio=0, color="green")
                    ax.quiver(_poses[:,0], _poses[:,1], _poses[:,2], _axes_z[:, 0]*axis_len, _axes_z[:, 1]*axis_len, _axes_z[:, 2]*axis_len, linewidths=-.5, arrow_length_ratio=0, color="blue")
            else:
                if not clear:
                    c = None
                ax.scatter(_poses[:,0], _poses[:,1], color=color,marker=marker,label=f"{title} agent {i}")
        
        if plot3d and show_edges:
            # print(edges_a, edges_b)
            if len(edges_a) > 0:
                ax.quiver(edges_a[:,0],edges_a[:,1],edges_a[:,2],edges_b[:,0],edges_b[:,1],edges_b[:,2], arrow_length_ratio=0, linewidths=-.5, color=edge_color)
                if show_raw_edges:
                    ax.quiver(edges_a[:,0],edges_a[:,1],edges_a[:,2],edge_real[:,0],edge_real[:,1],edge_real[:,2], linewidths=-.5, color="black")
        elif show_edges:
            ax.quiver(edges_a[:,0],edges_a[:,1],edges_b[:,0],edges_b[:,1], headlength=0, headaxislength=0, scale_units="xy", linewidths=-.5, color=edge_color)
            if show_raw_edges:
                ax.quiver(edges_a[:,0],edges_a[:,1],edge_real[:,0],edge_real[:,1], linewidths=-.5, color="black")
        ax.grid(True)   
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{title}.png")
        return ax
    
    def align_with_pose_id(self, _id):
        pos = copy.copy(self.keyframes[_id].pos)
        quat = copy.copy(self.keyframes[_id].quat)
        _inv_quat = quaternion_inverse(quat)
        _inv_pos = quaternion_rotate(_inv_quat, -pos)

        for _id in self.keyframes:
            kf = self.keyframes[_id]
            kf.pos = quaternion_rotate(_inv_quat, kf.pos + _inv_pos)
            kf.quat = quaternion_multiply(_inv_quat, kf.quat)

    def read_txt(self, fname, agent_id=-1, update_only = False):
        with open(fname) as f:
            content = f.read()
        vertices_matched = re.findall("(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", content, flags=0)
        if not update_only:
            agent = self.agents[agent_id]
        for vertex in vertices_matched:
            pos = np.array([float(vertex[1]), float(vertex[2]),  float(vertex[3])])
            quat = np.array([float(vertex[7]), float(vertex[4]),  float(vertex[5]),  float(vertex[6])]) # WXYZ
            if update_only:
                self.keyframes[int(vertex[0])].pos = pos
                self.keyframes[int(vertex[0])].quat = quat
            else:
                kf = KeyFrame(int(vertex[0]),  agent_id, pos, quat)
                self.keyframes[int(vertex[0])] = kf
                agent.add_keyframe(kf)


    def read_g2o(self, fname, agent_id=-1, update_only = False, cvt_id=False):
        #print(f"Read from g2o  file {fname}")
        with open(fname) as f:
            content = f.read()
        vertices_matched = re.findall("VERTEX_SE3:QUAT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",
                content, flags=0)
        if not update_only:
            agent = self.agents[agent_id]
        for vertex in vertices_matched:
            v_id = int(vertex[0])
            _, _id = extrack_keyframe_id(v_id)
            pos = np.array([float(vertex[1]), float(vertex[2]),  float(vertex[3])])
            quat = np.array([float(vertex[7]), float(vertex[4]),  float(vertex[5]),  float(vertex[6])]) # WXYZ
            if update_only:
                if _id in self.keyframes:
                    self.keyframes[_id].pos = pos
                    self.keyframes[_id].quat = quat
            else:
                if _id in self.keyframes:
                    continue
                kf = KeyFrame(_id,  agent_id, pos, quat)
                self.add_keyframe(kf, agent_id)
        vertices_2d_matched = re.findall("VERTEX_SE2 (\\S+) (\\S+) (\\S+) (\\S+)", content, flags=0)
        for vertex in vertices_2d_matched:
            pos = np.array([float(vertex[1]), float(vertex[2]), 0.0])
            quat = quaternion_from_euler(0, 0, float(vertex[3])) # XYZW
            v_id = int(vertex[0])
            _, _id = extrack_keyframe_id(v_id)
            if update_only and _id in self.keyframes:
                self.keyframes[_id].pos = pos
                self.keyframes[_id].quat = quat
            else:
                if _id in self.keyframes:
                    continue
                kf = KeyFrame(_id,  agent_id, pos, quat)
                self.add_keyframe(kf, agent_id)
        if update_only:
            return
        # edges_matched = re.findall("EDGE_SE3:QUAT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", content, flags=0)
        edges_matched = re.findall("EDGE_SE3:QUAT\s+([ ,\S]+)\n", content, flags=0)
        for edge_str in edges_matched:
            edge = edge_str.split()
            # print(edge, len(edge))
            pos = np.array([float(edge[2]), float(edge[3]),  float(edge[4])])
            quat = np.array([float(edge[8]), float(edge[5]),  float(edge[6]),  float(edge[7])]) # WXYZ
            v_ida = int(edge[0])
            _, _ida = extrack_keyframe_id(v_ida)
            v_idb = int(edge[1])
            _, _idb = extrack_keyframe_id(v_idb)

            is_inter = True
            if agent.has_keyframe(_ida) and agent.has_keyframe(_idb):
                is_inter = False
            
            _inf_mat = np.zeros((6, 6))
            c = 9
            for i in range(6):
                for j in range(i, 6):
                    _inf_mat[i, j] = math.fabs(float(edge[c]))
                    if i!=j:
                        _inf_mat[j, i] = _inf_mat[i, j]
                    c += 1

            _edge = Edge(_ida, _idb, pos, quat,is_inter, inf_mat=_inf_mat)
            self.add_edge(_edge)

        edges_2d_matched = re.findall("EDGE_SE2\s+([ ,\S]+)\n", content, flags=0)
        for edge_str in edges_2d_matched:
            edge = edge_str.split()
            pos = np.array([float(edge[2]), float(edge[3]), 0.0])
            quat = quaternion_from_euler(0, 0, float(edge[4])) # XYZW
            is_inter = True
            v_ida = int(edge[0])
            _, _ida = extrack_keyframe_id(v_ida)
            v_idb = int(edge[1])
            _, _idb = extrack_keyframe_id(v_idb)

            if agent.has_keyframe(_ida) and agent.has_keyframe(_idb):
                is_inter = False

            _inf_mat = np.zeros((3, 3))
            c = 5
            for i in range(3):
                for j in range(i, 3):
                    _inf_mat[i, j] = math.fabs(float(edge[c]))
                    c += 1
                    if i!=j:
                        _inf_mat[j, i] = _inf_mat[i, j]
            inf_mat = np.zeros((6, 6))
            inf_mat[0:2, 0:2] = _inf_mat[0:2, 0:2]
            inf_mat[2, 2] = _inf_mat[0, 0]*10

            inf_mat[3, 3] = _inf_mat[2, 2]
            inf_mat[4, 4] = _inf_mat[2, 2]
            inf_mat[5, 5] = _inf_mat[2, 2]

            _edge = Edge(_ida, _idb, pos, quat,is_inter,inf_mat=inf_mat)
            self.add_edge(_edge)
            
            if not update_only:
                self.inter_edge_num += agent.inter_edge_num

    def read_g2o_folder(self, folder, read_optimized=False, prt=True, update_only=False):
        max_keyframes = 0
        min_keyframes = 10000
        
        for file in os.listdir(folder):
            if file.endswith(".g2o"):
                agent_id = -1
                if read_optimized:
                    filename = os.path.splitext(file)[0]
                    names = filename.split("_")
                    if len(names) > 1 and names[1] == "optimized":
                        try:
                            agent_id = int(names[0])
                        except:
                            print("Give up", file)
                else:
                    try:
                        agent_id = int(os. path. splitext(file)[0])
                    except:
                        pass
                        # print("Give up", file)
                if agent_id >= 0:
                    # print(f"Read file {file}")
                    if not update_only:
                        self.agents[agent_id] = Agent(agent_id)
                    fpath = os.path.join(folder, file)
                    self.read_g2o(fpath, agent_id, update_only=update_only)
                    keyframe_num = len(self.agents[agent_id].keyframes)
                    if  keyframe_num > max_keyframes:
                        max_keyframes = keyframe_num
                    
                    if  keyframe_num < min_keyframes:
                        min_keyframes = keyframe_num
                
        _, inter_edge = self.update_edges()
        if prt:
            print(f"Total agents {len(self.agents)} keyframes {len(self.keyframes)} edges {len(self.edges)} inter edge {inter_edge} comm_vol {self.communication_volume()} keyframes {min_keyframes}<->{max_keyframes}")
    def evaluate_trajectory_length(self):
        # self.update_edges()
        total_length = 0
        for agent in self.agents.values():
            total_length += agent.evaluate_trajectory_length()
        return total_length/len(self.agents)

    def rename_keyframes_by_index(self):
        for kf_id in self.keyframes:
            self.keyframes[kf_id].keyframe_id = self.id2index[kf_id]
        for edge in self.edges:
            edge.keyframe_ida = self.id2index[edge.keyframe_ida]
            edge.keyframe_idb = self.id2index[edge.keyframe_idb]

    def rename_keyframes_from_index(self):
        for kf_id in self.keyframes:
            self.keyframes[kf_id].keyframe_id = kf_id
        for edge in self.edges:
            edge.keyframe_ida = self.index2id[edge.keyframe_ida]
            edge.keyframe_idb = self.index2id[edge.keyframe_idb]

    def write_to_g2o_folder(self, path, cvt_id=False, duplicate_inter_edge=True, update_edges=True):
        #duplicate_inter_edge True for DGS. False for DSLAM
        import shutil
        from pathlib import Path
        try:
            shutil.rmtree(path)
        except:
            pass
        Path(path).mkdir(parents=True, exist_ok=True)
        if update_edges:
            self.update_edges(duplicate_inter_edge=duplicate_inter_edge)
        c = 0
        for agent_id in self.agents:
            c += len(self.agents[agent_id].edges)
            self.agents[agent_id].write_to_g2o(f"{path}/{agent_id}.g2o", cvt_id)
        # print(f"Wrote {c} edges total {len(self.edges)}")

    def write_to_g2o(self, path, cvt_id=False, agent_id=-1):
        if agent_id < 0:
            for agent_id in self.agents:
                self.agents[agent_id].write_to_g2o(path, cvt_id, add=True)
        else:
            self.agents[agent_id].write_to_g2o(path, cvt_id)

    def write_to_csv(self, output_path, frame_id_to_stamp=None):
        for i in self.agents:
            self.agents[i].write_to_csv(f"{output_path}/pgo_{i}.csv", frame_id_to_stamp)
    
    def read_g2o_single(self, path, update_only=False, cvt_id=False, verbose=False):
        if not update_only:
            self.agents[0] = Agent(0)
        self.read_g2o(path, 0, update_only=update_only, cvt_id=cvt_id)
        if  not update_only:
            _, inter_edge = self.update_edges()
            if verbose:
                print(f"Total agents {len(self.agents)} keyframes {len(self.keyframes)} edges {len(self.edges)} inter edge {inter_edge} \
comm_vol {self.communication_volume()} from path {path}")

    def read_sesync_poses(self, path):
        self.agents[0] = Agent(0)
        poses_ = np.loadtxt(path)
        cols = poses_.shape[1]
        for i in range(cols//4):
            t = poses_[:,i]
            R = np.eye(4)
            R[:3,:3] =  poses_[:, i+cols//4:i + cols//4 + 3]
            kf = KeyFrame(i, 0, t, quaternion_from_matrix(R))
            self.keyframes[i] = kf
            self.agents[0].add_keyframe(kf)

        print(f"Total agents {len(self.agents)} keyframes {len(self.keyframes)} edges {len(self.edges)} comm_vol {self.communication_volume()}")

    def read_txt_single(self, path):
        self.agents[0] = Agent(0)
        self.read_txt(path, 0)
        print(f"Total agents {len(self.agents)} keyframes {len(self.keyframes)} edges {len(self.edges)} comm_vol {self.communication_volume()}")
    
    def convert_to_4dof(self):
        #Convert all edges and keyframes to 4DoF
        for edge in self.edges:
            rel_pos_6d = edge.pos
            rel_quat_6d = edge.quat
            kf_a = self.keyframes[edge.keyframe_ida]
            pos_a = kf_a.pos
            quat_a = kf_a.quat
            pos_b = pos_a + quaternion_rotate(quat_a, rel_pos_6d)
            dyaw, _, _ = quat2eulers(rel_quat_6d)
            yawa, _, _ = quat2eulers(quat_a)
            dpos = quaternion_rotate(quaternion_from_euler(0, 0, -yawa) , pos_b - pos_a)
            edge.pos = dpos
            edge.quat = quaternion_from_euler(0, 0, dyaw)

        for kf_id in self.keyframes:
            kf = self.keyframes[kf_id]
            yaw, _, _ = quat2eulers(kf.quat)
            kf.quat = quaternion_from_euler(0, 0, yaw)
    
    def evaluate_cost(self):
        cost = 0
        for edge in self.edges:
            kf_a = self.keyframes[edge.keyframe_ida]
            kf_b = self.keyframes[edge.keyframe_idb]
            rel_pos_6d = edge.pos
            rel_quat_6d = edge.quat
            pos_a = kf_a.pos
            quat_a = kf_a.quat
            pos_b = pos_a + quaternion_rotate(quat_a, rel_pos_6d)
            quat_b = quaternion_multiply(quat_a, rel_quat_6d)
            dq = quaternion_multiply(quat_b, quaternion_inverse(kf_b.quat))
            err = np.concatenate([pos_b - kf_b.pos, 2*dq[1:]])
            _cost = 0.5 * err @ edge.information_matrix @ err.T
            cost += _cost
        return cost
    
if __name__ == '__main__':
    pg = PoseGraph()
    pg.read_g2o_folder("/home/xuhao/data/pose_graph/example_2robots")
    pg.show()
    pg.partitioning(True)
    pg.show("pose-graph-re")