#include "swarm_outlier_rejection.hpp"

#include <d2common/d2basetypes.h>
#include <stdio.h>

#include <d2common/utils.hpp>
#include <fstream>

#include "fast_max-clique_finder/src/findClique.h"
#include "fast_max-clique_finder/src/graphIO.h"

using D2Common::FrameIdType;
using D2Common::Utility::TicToc;

#define PCM_DEBUG_OUTPUT

namespace D2PGO {
std::fstream pcm_errors;
FILE *f_logs;
SwarmLocalOutlierRejection::SwarmLocalOutlierRejection(
    int _self_id, const SwarmLocalOutlierRejectionParams &_param,
    std::map<int, Swarm::DroneTrajectory> &_ego_motion_trajs)
    : self_id(_self_id), param(_param), ego_motion_trajs(_ego_motion_trajs) {
  if (param.debug_write_pcm_errors) {
    f_logs = fopen("/root/output/pcm_logs.txt", "w");
    pcm_errors.open("/root/output/pcm_errors.txt", std::ios::out);
    pcm_errors.close();
    fclose(f_logs);
  }
}

std::vector<int64_t> SwarmLocalOutlierRejection::good_loops() {
  lcm_mutex.lock();
  std::vector<int64_t> ret;
  for (auto &it1 : good_loops_set) {
    for (auto &it2 : it1.second) {
      for (auto it3 : it2.second) {
        ret.push_back(it3);
      }
    }
  }
  lcm_mutex.unlock();
  return ret;
}

std::vector<Swarm::LoopEdge>
SwarmLocalOutlierRejection::OutlierRejectionLoopEdges(
    ros::Time stamp, const std::vector<Swarm::LoopEdge> &available_loops) {
  if (param.debug_write_pcm_errors) {
    pcm_errors.open("/root/output/pcm_errors.txt", std::ios::app);
    f_logs = fopen("/root/output/pcm_logs.txt", "a");
  }

  std::map<int, std::map<int, std::vector<Swarm::LoopEdge>>> new_loops;
  std::vector<Swarm::LoopEdge> good_loops;
  int new_loop_count = 0;
  for (auto &edge : available_loops) {
    if (all_loops_set.find(edge.id) == all_loops_set.end()) {
      new_loops[edge.id_a][edge.id_b].emplace_back(edge);
      if (edge.id_a != edge.id_b) {
        new_loops[edge.id_b][edge.id_a].emplace_back(edge);
      }
      if (all_loop_map.find(edge.id) == all_loop_map.end()) {
        all_loop_map[edge.id] = edge;
      }
      all_loops_set_by_pair[edge.id_a][edge.id_b].insert(edge.id);
      all_loops_set_by_pair[edge.id_b][edge.id_a].insert(edge.id);
      new_loop_count += 1;
    }
    all_loops_set.insert(edge.id);
  }
  SPDLOG_INFO("(OutlierRejection) {} new loops, {} total loops",
         new_loop_count, all_loops_set.size());

  if (param.redundant) {
    // Note in D2SLAM this is on because we do not broadcast the loops.
    // So it's natural to be distributed.
    // the other branch is for debugging only.
    for (auto it_a : new_loops) {
      for (auto it_b : it_a.second) {
        if (it_a.first >= it_b.first) {
          OutlierRejectionLoopEdgesPCM(it_b.second, it_a.first, it_b.first);
        }
      }
    }
  } else {
    for (auto it_a : new_loops) {
      for (auto it_b : it_a.second) {
        if (it_a.first == self_id) {
          OutlierRejectionLoopEdgesPCM(it_b.second, it_a.first, it_b.first);
        }
      }
    }
  }

  lcm_mutex.lock();
  for (auto &loop : available_loops) {
    auto id_a = loop.id_a;
    auto id_b = loop.id_b;
    if (good_loops_set.find(id_a) == good_loops_set.end() ||
        good_loops_set[id_a].find(id_b) == good_loops_set[id_a].end()) {
      // The inlier set of the pair in good loop not established, so we make use
      // all of them
      good_loops.emplace_back(loop);
    } else {
      auto _good_loops_set = good_loops_set[loop.id_a][loop.id_b];
      if (_good_loops_set.find(loop.id) != _good_loops_set.end()) {
        good_loops.emplace_back(loop);
      }
    }
  }
  lcm_mutex.unlock();

  if (param.debug_write_pcm_errors) {
    pcm_errors.close();
    fclose(f_logs);
  }

  return good_loops;
}

void SwarmLocalOutlierRejection::OutlierRejectionLoopEdgesPCM(
    const std::vector<Swarm::LoopEdge> &new_loops, int id_a, int id_b) {
  std::map<FrameIdType, int> bad_pair_count;

  auto &pcm_graph = loop_pcm_graph[id_a][id_b];
  auto &_all_loops = all_loops[id_a][id_b];

  TicToc tic1;

  for (size_t i = 0; i < new_loops.size(); i++) {
    auto &edge1 = new_loops[i];
    // Now only process inter-edges
    while (pcm_graph.size() < _all_loops.size() + 1) {
      pcm_graph.emplace_back(std::vector<int>(0));
    }

    auto p_edge1 = edge1.relative_pose;
    Matrix6d _cov_mat_1 = edge1.getCovariance();

    for (size_t j = 0; j < _all_loops.size(); j++) {
      auto &edge2 = _all_loops[j];
      auto _cov_mat_2 = edge2.getCovariance();
      ;
      Matrix6d _covariance = _cov_mat_1 + _cov_mat_2;

      int same_robot_pair = edge2.same_robot_pair(edge1);
      if (same_robot_pair > 0) {
        // Now we can compute the consistency error.
        std::pair<Swarm::Pose, Matrix6d> odom_a, odom_b;
        Swarm::Pose p_edge2;
        double traj_a = 0, traj_b = 0;

        if (same_robot_pair == 1) {
          p_edge2 = edge2.relative_pose;
          // ODOM is tsa->tsb
          odom_a =
              ego_motion_trajs.at(edge1.id_a)
                  .get_relative_pose_by_frame_id(
                      edge1.keyframe_id_a, edge2.keyframe_id_a, param.is_4dof);
          odom_b =
              ego_motion_trajs.at(edge1.id_b)
                  .get_relative_pose_by_frame_id(
                      edge1.keyframe_id_b, edge2.keyframe_id_b, param.is_4dof);
          if (param.debug_write_debug) {
            traj_a = ego_motion_trajs.at(edge1.id_a)
                         .trajectory_length_by_ts(edge1.ts_a, edge2.ts_a);
            traj_b = ego_motion_trajs.at(edge1.id_b)
                         .trajectory_length_by_ts(edge1.ts_b, edge2.ts_b);
          }

          _covariance += odom_a.second + odom_b.second;

        } else if (same_robot_pair == 2) {
          p_edge2 = edge2.relative_pose.inverse();
          odom_a =
              ego_motion_trajs.at(edge1.id_a)
                  .get_relative_pose_by_frame_id(
                      edge1.keyframe_id_a, edge2.keyframe_id_b, param.is_4dof);
          odom_b =
              ego_motion_trajs.at(edge1.id_b)
                  .get_relative_pose_by_frame_id(
                      edge1.keyframe_id_b, edge2.keyframe_id_a, param.is_4dof);

          if (param.debug_write_debug) {
            traj_a = ego_motion_trajs.at(edge1.id_a)
                         .trajectory_length_by_ts(edge1.ts_a, edge2.ts_b);
            traj_b = ego_motion_trajs.at(edge1.id_b)
                         .trajectory_length_by_ts(edge1.ts_b, edge2.ts_a);
          }

          _covariance += odom_a.second + odom_b.second;
        }

        Swarm::Pose err_pose =
            odom_a.first * p_edge2 * odom_b.first.inverse() * p_edge1.inverse();
        auto logmap = err_pose.log_map();
        double smd =
            Swarm::computeSquaredMahalanobisDistance(logmap, _covariance);

        if (smd < param.pcm_thres) {
          // Add edge i to j
          pcm_graph[_all_loops.size()].push_back(j);
          pcm_graph[j].push_back(_all_loops.size());
        }

        if (param.debug_write_debug) {
          fprintf(f_logs, "\n");
          fprintf(f_logs, "EdgePair %ld->%ld\n", edge1.id, edge2.id);
          fprintf(f_logs,
                  "Edge1 %ld->%ld DOF %d Pose %s cov_1 "
                  "[%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e]\n",
                  edge1.keyframe_id_a, edge1.keyframe_id_b, edge1.res_count,
                  edge1.relative_pose.toStr().c_str(), _cov_mat_1(0, 0),
                  _cov_mat_1(1, 1), _cov_mat_1(2, 2), _cov_mat_1(3, 3),
                  _cov_mat_1(4, 4), _cov_mat_1(5, 5));
          fprintf(f_logs,
                  "Edge2 %ld->%ld DOF %d Pose %s cov_2 "
                  "[%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e]\n",
                  edge2.keyframe_id_a, edge2.keyframe_id_b, edge2.res_count,
                  edge2.relative_pose.toStr().c_str(), _cov_mat_2(0, 0),
                  _cov_mat_2(1, 1), _cov_mat_2(2, 2), _cov_mat_2(3, 3),
                  _cov_mat_2(4, 4), _cov_mat_2(5, 5));

          auto cov = odom_a.second;
          fprintf(f_logs,
                  "odom_a %s traj len %.2f cov (T, Q) "
                  "[%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e]\n",
                  odom_a.first.toStr().c_str(), traj_a, cov(0, 0), cov(1, 1),
                  cov(2, 2), cov(3, 3), cov(4, 4), cov(5, 5));
          cov = odom_b.second;
          fprintf(f_logs,
                  "odom_b %s traj len %.2f cov (T, Q) "
                  "[%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e]\n",
                  odom_b.first.toStr().c_str(), traj_b, cov(0, 0), cov(1, 1),
                  cov(2, 2), cov(3, 3), cov(4, 4), cov(5, 5));
          fprintf(f_logs,
                  "err_pose %s logmap "
                  "[%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e]\n",
                  err_pose.toStr().c_str(), logmap(0), logmap(1), logmap(2),
                  logmap(3), logmap(4), logmap(5));
          fprintf(f_logs,
                  "squaredMahalanobisDistance %f Same Direction %d _cov(T, Q)  "
                  "[%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e,%+3.1e]\n",
                  smd, same_robot_pair == 1, _covariance(0, 0),
                  _covariance(1, 1), _covariance(2, 2), _covariance(3, 3),
                  _covariance(4, 4), _covariance(5, 5));
        }

        if (param.debug_write_pcm_errors) {
          pcm_errors << edge1.id << " " << edge2.id << " " << smd << " "
                     << std::endl;
        }
      }
    }
    _all_loops.push_back(edge1);
  }

  double compute_pcm_erros = tic1.toc();
  std::vector<int> max_clique_data;
  FMC::CGraphIO pcm_graph_fmc;
  pcm_graph_fmc.m_vi_Vertices.push_back(0);
  for (size_t i = 0; i < pcm_graph.size(); i++) {
    pcm_graph_fmc.m_vi_Edges.insert(pcm_graph_fmc.m_vi_Edges.end(),
                                    pcm_graph[i].begin(), pcm_graph[i].end());
    pcm_graph_fmc.m_vi_Vertices.push_back(pcm_graph_fmc.m_vi_Edges.size());
  }
  pcm_graph_fmc.CalculateVertexDegrees();
  if (param.incremental_pcm) {
    TicToc tic;
    int prev_max_clique_size = good_loops_set[id_a][id_b].size();
    int ret = FMC::maxCliqueHeuIncremental(
        pcm_graph_fmc, new_loops.size(), prev_max_clique_size, max_clique_data);
    if (ret > 0 && max_clique_data.size() > 0) {
      good_loops_set[id_a][id_b].clear();
      good_loops_set[id_b][id_a].clear();
      for (auto i : max_clique_data) {
        good_loops_set[id_a][id_b].insert(_all_loops[i].id);
        good_loops_set[id_b][id_a].insert(_all_loops[i].id);
      }
    }
    SPDLOG_INFO(
        "(OutlierRejection) {}<->{} compute_pcm_errors {:.1f}ms "
        "maxCliqueHeuInc takes {:.1f}ms ret {}({}) loops {} good {}",
        id_a, id_b, compute_pcm_erros, tic.toc(), ret, max_clique_data.size(),
        _all_loops.size(), good_loops_set[id_a][id_b].size());
  } else {
    TicToc tic;
    FMC::maxCliqueHeu(pcm_graph_fmc, max_clique_data);
    SPDLOG_INFO(
        "(OutlierRejection) {}<->{} compute_pcm_errors {:.1f}ms "
        "maxCliqueHeu takes {:.1f}ms loops {} good {}",
        id_a, id_b, compute_pcm_erros, tic.toc(), _all_loops.size(),
        max_clique_data.size());
    // In non-incremental mode, we need to clear the good_loops_set
    good_loops_set[id_a][id_b].clear();
    good_loops_set[id_b][id_a].clear();
    for (auto i : max_clique_data) {
      good_loops_set[id_a][id_b].insert(_all_loops[i].id);
      good_loops_set[id_b][id_a].insert(_all_loops[i].id);
    }
  }
}
}  // namespace D2PGO