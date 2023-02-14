#pragma once
#include <iostream>
#include <thread>
#include <mutex>
#include <swarm_msgs/drone_trajectory.hpp>
#include <swarm_msgs/relative_measurments.hpp>
#include "../d2pgo_config.h"

namespace D2PGO {

typedef std::vector<std::vector<int>> DisjointGraph;
class SwarmLocalOutlierRejection {
    SwarmLocalOutlierRejectionParams param;
    std::map<int, Swarm::DroneTrajectory>  & ego_motion_trajs;
    //Drone  ida           idb            index_det       linked dets
    std::map<int, std::map<int, DisjointGraph>> loop_pcm_graph;
    std::map<int, std::map<int, std::vector<Swarm::LoopEdge>>> all_loops;
    std::set<int64_t> all_loops_set;

    void OutlierRejectionLoopEdgesPCM(const std::vector<Swarm::LoopEdge > & inter_loops, int id_a, int id_b);
    std::vector<int64_t> good_loops();
public:
    std::map<int, std::map<int, std::set<int64_t>>> all_loops_set_by_pair;
    std::map<int, std::map<int, std::set<int64_t>>> good_loops_set;
    std::map<int64_t, Swarm::LoopEdge> all_loop_map;
    int self_id = -1;

    std::mutex lcm_mutex;
    
    SwarmLocalOutlierRejection(int self_id, const SwarmLocalOutlierRejectionParams &_param, std::map<int, Swarm::DroneTrajectory> &_ego_motion_trajs);
    std::vector<Swarm::LoopEdge> OutlierRejectionLoopEdges(ros::Time stamp, const std::vector<Swarm::LoopEdge> & available_loops);
};
}