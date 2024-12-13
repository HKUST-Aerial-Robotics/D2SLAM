#include <nav_msgs/Path.h>
#include <std_msgs/Int32.h>

#include <thread>

#include "../src/d2pgo.h"
#include "d2pgo_test.hpp"
#include "posegraph_g2o.hpp"

using namespace D2PGO;

void testDummy() {
  D2PGOConfig config;
  config.self_id = 0;
  config.main_id = 0;
  config.loop_distance_threshold = 10000;
  D2PGO::D2PGO* pgo = new D2PGO::D2PGO(config);
  auto frame0 = std::make_shared<D2BaseFrame>(0.0, 0, 0, 0, true, Swarm::Pose::Identity());
  auto frame1 = std::make_shared<D2BaseFrame>(1.0, 1, 0, 0, true,
                     Swarm::Pose(Vector3d(1, 0, 0), Quaterniond(1, 0, 0, 0)));
  auto frame2 = std::make_shared<D2BaseFrame>(2.0, 2, 0, 0, true,
                     Swarm::Pose(Vector3d(2, 0, 0), Quaterniond(1, 0, 0, 0)));
  Swarm::LoopEdge edge(0, 1,
                       Swarm::Pose(Vector3d(1, 0, 0), Quaterniond(1, 0, 0, 0)),
                       Eigen::Matrix6d::Identity());
  Swarm::LoopEdge edge1(1, 2,
                        Swarm::Pose(Vector3d(1, 0, 0), Quaterniond(1, 0, 0, 0)),
                        Eigen::Matrix6d::Identity());
  Swarm::LoopEdge edge2(0, 2,
                        Swarm::Pose(Vector3d(2, 0, 0), Quaterniond(1, 0, 0, 0)),
                        Eigen::Matrix6d::Identity());
  std::vector<Swarm::LoopEdge> loops{edge, edge1, edge2};
  pgo->addFrame(frame0);
  pgo->addFrame(frame1);
  pgo->addFrame(frame2);
  pgo->addLoop(edge, false);
  pgo->addLoop(edge1, false);
  pgo->addLoop(edge2, false);
  pgo->rotInitial(loops);
}

int main(int argc, char** argv) { testDummy(); }