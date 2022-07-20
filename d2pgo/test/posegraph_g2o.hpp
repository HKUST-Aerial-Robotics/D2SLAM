#pragma once
#include <boost/filesystem.hpp>
#include <swarm_msgs/Pose.h>
#include <swarm_msgs/relative_measurments.hpp>
#include <d2common/d2vinsframe.h>
#include <regex>

#define POSE_SIZE_4DOF 4
#define POSE_SIZE_6DOF 7

#define RESIDUAL_SIZE_4DOF 4
#define RESIDUAL_SIZE_6DOF 6

using namespace D2Common;
namespace D2PGO {

struct G2oParseParam {
    int agents_num = 0;
    bool is_4dof = true;
    std::string g2o_path = "";
    std::string output_path = "";
    bool silent = false;
};

void read_g2o_agent(
    std::string path,
    std::map<FrameIdType, D2BaseFrame> & keyframeid_agent_pose,
    std::vector<Swarm::LoopEdge> & edges,
    bool is_4dof, int drone_id=-1);

void read_g2o_multi_agents(
    std::string path,
    std::map<int, std::map<FrameIdType, D2BaseFrame>> & keyframeid_agent_pose,
    std::map<int, std::vector<Swarm::LoopEdge>> & edges,
    G2oParseParam param
);

void write_result_to_g2o(
    std::string path,
    std::vector<D2BaseFrame> states,
    bool is_4dof
);
}