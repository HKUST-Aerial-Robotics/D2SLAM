#include "posegraph_g2o.hpp"

#include <fstream>
#include <random>

namespace fs = boost::filesystem;

namespace D2PGO {

std::regex reg_vertex_se3(
    "VERTEX_SE3:QUAT\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(\\S+"
    ")\\s+(\\S+)\\s+(\\S+)");
std::regex reg_edge_se3("EDGE_SE3:QUAT\\s+([\\s,\\S]+)");

#define IDX_WITH_AGENT_ID_MIN 6989586621679009792

extern std::random_device rd;
extern std::default_random_engine eng;
extern std::normal_distribution<double> d;

bool is_number(const std::string& s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

std::pair<int, int64_t> extrackKeyframeId(const int64_t& input) {
  if (input < IDX_WITH_AGENT_ID_MIN) {
    return std::make_pair(0, input);
  }
  const int keyBits = 8 * 8;
  const int chrBits = 1 * 8;
  const int indexBits = keyBits - chrBits;
  const int64_t chrMask = (int64_t)255 << indexBits;
  const int64_t indexMask = ~chrMask;
  int agent_id = ((input & chrMask) >> indexBits) - 97;
  FrameIdType keyframe_id = input & indexMask;
  return std::make_pair(agent_id, keyframe_id);
}

std::vector<std::pair<int, std::string>> get_all(fs::path const& root,
                                                 std::string const& ext) {
  std::vector<std::pair<int, std::string>> paths;
  if (fs::exists(root) && fs::is_directory(root)) {
    for (auto const& entry : fs::directory_iterator(root)) {
      if (fs::is_regular_file(entry) && entry.path().extension() == ext &&
          is_number(entry.path().stem().string())) {
        // std::cout << "Filename: " << entry.path() << std::endl;
        paths.emplace_back(std::make_pair(
            std::stoi(entry.path().stem().string()), entry.path().string()));
      }
    }
  }
  return paths;
}

bool match_vertex_se3(std::string line, int& agent_id, FrameIdType& kf_id,
                      Swarm::Pose& pose, int max_agent_id) {
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
    auto ret = extrackKeyframeId(std::stoll(sm[1].str()));
    kf_id = ret.second;
    agent_id = ret.first;
    if (agent_id > max_agent_id) {
      return false;
    }

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

bool match_edge_se3(std::string line, int& agent_ida, FrameIdType& ida,
                    int& agent_idb, FrameIdType& idb, Swarm::Pose& pose,
                    Eigen::Matrix6d& information, int max_agent_id) {
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
    int64_t tmp_a, tmp_b;
    stream >> tmp_a >> tmp_b;
    auto ret_a = extrackKeyframeId(tmp_a);
    agent_ida = ret_a.first;
    ida = ret_a.second;
    auto ret_b = extrackKeyframeId(tmp_b);
    agent_idb = ret_b.first;
    idb = ret_b.second;
    if (agent_ida > max_agent_id || agent_idb > max_agent_id) {
      return false;
    }
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

void read_g2o_agent(std::string path,
                    std::map<FrameIdType, D2BaseFramePtr>& keyframeid_agent_pose,
                    std::vector<Swarm::LoopEdge>& edges, bool is_4dof,
                    int max_agent_id, int drone_id, bool ignore_infor) {
  std::ifstream infile(path);
  std::string line;
  while (std::getline(infile, line)) {
    // std::cout << "line^" << line << "$" << std::endl;
    Swarm::Pose pose;
    FrameIdType id_a, id_b;
    int agent_id;
    auto success = match_vertex_se3(line, agent_id, id_a, pose, max_agent_id);
    if (success) {
      // Add new vertex here
      D2BaseFramePtr frame = std::make_shared<D2BaseFrame>();
      frame->drone_id = agent_id;
      frame->odom.pose() = pose;
      frame->initial_ego_pose = pose;
      frame->frame_id = id_a;
      frame->reference_frame_id = 0;
      keyframeid_agent_pose[id_a] = frame;
    } else {
      Eigen::Matrix6d information;
      int agent_id_b;
      success = match_edge_se3(line, agent_id, id_a, agent_id_b, id_b, pose,
                               information, max_agent_id);
      if (ignore_infor) {
        information = Eigen::Matrix6d::Identity();
      }
      if (success) {
        Swarm::LoopEdge edge(id_a, id_b, pose, information);
        edge.id_a = agent_id;
        edge.id_b = agent_id_b;
        edges.emplace_back(edge);
        // std::cout << "line" << line << std::endl;
        // printf("Edge drone %d->%d frame %ld->%ld\n", agent_id, agent_id_b,
        // id_a, id_b); std::cout << "information:\n" << information <<
        // std::endl;
      }
    }
  }
}

void read_g2o_multi_agents(
    std::string path,
    std::map<int, std::map<FrameIdType, D2BaseFramePtr>>& keyframeid_agent_pose,
    std::map<int, std::vector<Swarm::LoopEdge>>& edges, G2oParseParam param) {
  auto files = get_all(path, ".g2o");
  std::sort(files.begin(), files.end());
  int agent_num = files.size();
  for (unsigned int i = 0; i < files.size(); i++) {
    if (i >= param.agents_num) {
      break;
    }
    auto file = files[i].second;
    keyframeid_agent_pose[i] = std::map<FrameIdType, D2BaseFramePtr>();
    edges[i] = std::vector<Swarm::LoopEdge>();
    read_g2o_agent(file, keyframeid_agent_pose[i], edges[i], param.is_4dof,
                   param.agents_num - 1);
  }
  printf("g2o files %ld in path %s keyframe: %ld edges %ld\n", files.size(),
         path.c_str(), keyframeid_agent_pose.size(), edges.size());
}

void write_result_to_g2o(const std::string& path,
                         const std::vector<D2BaseFramePtr>& frames,
                         const std::vector<Swarm::LoopEdge>& edges,
                         bool write_ego_pose) {
  std::fstream file;
  file.open(path.c_str(), std::fstream::out);
  for (auto& frame : frames) {
    Swarm::Pose pose;
    if (write_ego_pose) {
      pose = frame->initial_ego_pose;
    } else {
      pose = frame->odom.pose();
    }
    auto quat = pose.att();
    auto pos = pose.pos();
    file << "VERTEX_SE3:QUAT " << frame->frame_id << " " << pos.x() << " "
         << pos.y() << " " << pos.z() << " ";
    file << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
         << std::endl;
  }

  for (auto& edge : edges) {
    Swarm::Pose pose = edge.relative_pose;
    auto info = edge.getInfoMat();
    auto quat = pose.att();
    auto pos = pose.pos();
    file << "EDGE_SE3:QUAT " << edge.keyframe_id_a << " " << edge.keyframe_id_b
         << " " << pos.x() << " " << pos.y() << " " << pos.z() << " ";
    file << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
         << " ";
    for (int i = 0; i < 6; ++i) {
      for (int j = i; j < 6; ++j) {
        file << info(i, j) << " ";
      }
    }
    file << std::endl;
  }
  file.close();
}
}  // namespace D2PGO