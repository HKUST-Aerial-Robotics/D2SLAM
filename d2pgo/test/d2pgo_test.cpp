#include "d2pgo_test.hpp"
#include <boost/program_options.hpp>
#include "posegraph_g2o.hpp"

using namespace D2PGO;

int main(int argc, char ** argv) {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("maxiter,i", po::value<int>()->default_value(10000), "number of max iterations")
        ("id", po::value<int>()->default_value(10000), "self id")
        ("cost,c", po::value<double>()->default_value(1e-4), "accept cost tolerance")
        ("solver,s", po::value<std::string>()->default_value("ARock"), "solver types, ceres, ARock")
        ("path,p", po::value<std::string>()->default_value(""), "Path of g2o file, if not speific, then use generated data")
        ("dof,d", po::value<int>()->default_value(4), "Default dof of solver")
        ("output,o", po::value<std::string>()->default_value("output.g2o"), "Path of output g2o file")
        ("silent", "Slient output");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    std::string g2o_path = vm["path"].as<std::string>();
    int drone_id = vm["id"].as<int>();
    if (g2o_path != "")
        printf("[D2PGO] agent %d parse g2o file: %s\n", drone_id, g2o_path.c_str());
    else
        printf("[D2PGO@%d] Need to indicate g2o path\n", drone_id);
    std::map<FrameIdType, D2BaseFrame> keyframeid_agent_pose;
    std::vector<Swarm::LoopEdge> edges;
    read_g2o_agent(g2o_path, keyframeid_agent_pose, edges, vm["dof"].as<int>() == 4);
    printf("[D2PGO@%d] Read %ld keyframes and %ld edges\n", drone_id, keyframeid_agent_pose.size(), edges.size());
}