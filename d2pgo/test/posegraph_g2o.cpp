#include "posegraph_g2o.hpp"
#include <fstream>
#include <random>

namespace fs = boost::filesystem;

namespace D2PGO {

std::regex reg_vertex_se3("VERTEX_SE3:QUAT\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)");
std::regex reg_edge_se3("EDGE_SE3:QUAT\\s+([\\s,\\S]+)");

extern std::random_device rd;
extern std::default_random_engine eng;
extern std::normal_distribution<double> d;

bool is_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(), 
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

std::vector<std::pair<int, std::string>>  get_all(fs::path const & root, std::string const & ext)
{
    std::vector<std::pair<int, std::string>>  paths;

    if (fs::exists(root) && fs::is_directory(root))
    {
        for (auto const & entry : fs::directory_iterator(root))
        {
            if (fs::is_regular_file(entry) && entry.path().extension() == ext && is_number(entry.path().stem().string())) 
            {
                // std::cout << "Filename: " << entry.path() << std::endl;
                paths.emplace_back(
                    std::make_pair(std::stoi(entry.path().stem().string()),
                        entry.path().string()));
            }
        }
    }

    return paths;
}   


bool match_vertex_se3(std::string line, uint64_t & kf_id, Swarm::Pose & pose) {
    Eigen::Vector3d pos;
    Eigen::Quaterniond quat;
    std::smatch sm;
    regex_search(line, sm, reg_vertex_se3);
    // std::cout << "The matches are:\n";
    // for( int i = 0 ; i < sm.size() ; ++i ){
    //     std::cout << i << ": [" << sm[i] << ']' << std::endl;
    // }
    // std::cout << std::endl;

    if (sm.size() > 8) {
        kf_id = std::stoll(sm[1].str());
        pos.x() = std::stod(sm[2].str());
        pos.y() = std::stod(sm[3].str());
        pos.z() = std::stod(sm[4].str());

        quat.x() = std::stod(sm[5].str());
        quat.y() = std::stod(sm[6].str());
        quat.z() = std::stod(sm[7].str());
        quat.w() = std::stod(sm[8].str());
        quat.normalize();
        pose = Swarm::Pose(pos, quat);
        return true;
    }
    return false;
}

bool match_edge_se3(std::string line, uint64_t & ida, uint64_t & idb, Swarm::Pose & pose, Eigen::Matrix6d & information) {
    Eigen::Vector3d pos;
    Eigen::Quaterniond quat;
    std::smatch sm;
    regex_search(line, sm, reg_edge_se3);
    // std::cout << "The matches are:\n";
    // for( int i = 0 ; i < sm.size() ; ++i ){
    //     std::cout << i << ": [" << sm[i] << ']' << std::endl;
    // }
    // std::cout << std::endl;
    if (sm.size() > 1) {
        std::stringstream stream(sm[1]);
        stream >> ida >> idb;
        stream >> pose;

        for (int i = 0; i < 6 && stream.good(); ++i) {
            for (int j = i; j < 6 && stream.good(); ++j) {
                stream >> information(i, j);
                if (i != j) {
                    information(j, i) = information(i, j);
                }
            }
        }
        return true;
    }

    return false;
}


void read_g2o_agent(
    std::string path,
    std::map<FrameIdType, D2BaseFrame> & keyframeid_agent_pose,
    std::vector<Swarm::LoopEdge> & edges,
    bool is_4dof) {

    std::ifstream infile(path);
    std::string line;
    std::vector<uint64_t> keyframeid_tmp;
    while (std::getline(infile, line))
    {
        // std::cout << "line^" << line << "$" << std::endl;
        Swarm::Pose pose;
        uint64_t id_a, id_b;
        auto success = match_vertex_se3(line, id_a, pose);
        if (success) {
            //Add new vertex here
            // std::cout << "Frame Id" << id_a << " Pos" << pos.transpose() << " quat" << quat.coeffs().transpose() << std::endl;
            D2BaseFrame frame;
            frame.odom.pose() = pose;
            frame.frame_id = id_a;
            keyframeid_agent_pose[id_a] = frame;
        } else {
            Eigen::Matrix6d information;
            success = match_edge_se3(line, id_a, id_b, pose, information);
            if (success) {
                // std::cout << "line" << line << std::endl;
                // std::cout << "Edge " << id_a << "-> " << id_b << " Pos" << pose.pos().transpose() << " quat" << pose.att().coeffs().transpose() << std::endl;
                // std::cout << "information:\n" << information << std::endl;
                Swarm::LoopEdge edge(id_a, id_b, pose, information);
                edges.emplace_back(edge);
            }
        }
    }
}

void read_g2o_multi_agents(
    std::string path,
    std::map<int, std::map<FrameIdType, D2BaseFrame>> & keyframeid_agent_pose,
    std::map<int, std::vector<Swarm::LoopEdge>> & edges,
    G2oParseParam param
) {
    auto files = get_all(path, ".g2o");
    std::sort(files.begin(), files.end());
    int agent_num = files.size();
    for (unsigned int i = 0; i < files.size(); i++) {
        auto file = files[i].second;
        keyframeid_agent_pose[i] = std::map<FrameIdType, D2BaseFrame>();
        edges[i] = std::vector<Swarm::LoopEdge>();
        read_g2o_agent(file, keyframeid_agent_pose[i], edges[i], param.is_4dof);
    }
    printf("g2o files %ld in path %s keyframe: %ld edges %ld\n", files.size(), path.c_str(), 
        keyframeid_agent_pose.size(), edges.size());
}
}