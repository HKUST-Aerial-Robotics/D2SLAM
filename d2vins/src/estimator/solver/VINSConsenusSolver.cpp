#include "VINSConsenusSolver.hpp"

#include "../../d2vins_params.hpp"
#include "../d2estimator.hpp"

namespace D2VINS {
void D2VINSConsensusSolver::setStateProperties() {
  estimator->setStateProperties();
}

void D2VINSConsensusSolver::broadcastData() {
  // sync pointers to remote_params.
  for (auto it : consenus_params) {
    auto pointer = it.first;
    auto consenus_param = it.second;
    if (!it.second.local_only) {
      remote_params[pointer] = std::map<int, VectorXd>();
      remote_params[pointer][self_id] = VectorXd(consenus_param.global_size);
      memcpy(remote_params.at(pointer).at(self_id).data(), pointer,
             consenus_param.global_size * sizeof(state_type));
      // printf("set to pose id %d\n", params[pointer].id);
      // std::cout << "remote_params[pointer][self_id]: " <<
      // remote_params[pointer][self_id].transpose() << std::endl;
    }
  }

  DistributedVinsData dist_data;
  for (auto it : all_estimating_params) {
    auto pointer = it.first;
    auto param = it.second;
    if (param.type == POSE) {
      dist_data.frame_ids.emplace_back(param.id);
      dist_data.frame_poses.emplace_back(Swarm::Pose(pointer));
    } else if (param.type == EXTRINSIC) {
      dist_data.cam_ids.emplace_back(param.id);
      dist_data.extrinsic.emplace_back(Swarm::Pose(pointer));
    }
  }
  dist_data.stamp = ros::Time::now().toSec();
  dist_data.drone_id = self_id;
  dist_data.solver_token = solver_token;
  dist_data.iteration_count = iteration_count;
  estimator->sendDistributedVinsData(dist_data);
}

void D2VINSConsensusSolver::receiveAll() {
  std::vector<DistributedVinsData> sync_datas = receiver->retrive_all();
  for (auto data : sync_datas) {
    updateWithDistributedVinsData(data);
  }
  if (params->verbose) {
    printf(
        "[ConsensusSolver::receiveAll@%d] token %d iteration %d receive "
        "finsish %ld/%ld\n",
        self_id, solver_token, iteration_count, sync_datas.size(),
        state->availableDrones().size());
  }
}

void D2VINSConsensusSolver::updateWithDistributedVinsData(
    const DistributedVinsData& dist_data) {
  auto _state = static_cast<D2EstimatorState*>(state);
  for (unsigned int i = 0; i < dist_data.frame_ids.size(); i++) {
    auto frame_id = dist_data.frame_ids[i];
    if (_state->hasFrame(frame_id)) {
      auto pointer = _state->getPoseState(frame_id);
      remote_params[pointer][dist_data.drone_id] = VectorXd(POSE_SIZE);
      dist_data.frame_poses[i].to_vector(
          remote_params[pointer][dist_data.drone_id].data());
      Swarm::Pose local(pointer);
    }
  }

  for (unsigned int i = 0; i < dist_data.cam_ids.size(); i++) {
    auto cam_id = dist_data.cam_ids[i];
    if (_state->hasCamera(cam_id)) {
      auto pointer = _state->getExtrinsicState(cam_id);
      remote_params[pointer][dist_data.drone_id] = VectorXd(POSE_SIZE);
      dist_data.extrinsic[i].to_vector(
          remote_params[pointer][dist_data.drone_id].data());
    }
  }

  if (config.verbose) {
    printf(
        "[ConsensusSolver::updateWithDistributedVinsData@%d] of drone %ld: "
        "solver token %ld iteration %ld\n",
        self_id, dist_data.drone_id, dist_data.solver_token,
        dist_data.iteration_count);
  }
}

void D2VINSConsensusSolver::waitForSync() {
  // Wait for all remote drone to publish result.
  Utility::TicToc tic;
  if (params->verbose) {
    printf("[ConsensusSolver::waitForSync@%d] token %d iteration %d\n", self_id,
           solver_token, iteration_count - 1);
  }
  std::vector<DistributedVinsData> sync_datas;
  while (tic.toc() < config.timout_wait_sync) {
    // Wait for remote data
    auto ret = receiver->retrive(solver_token, iteration_count);
    sync_datas.insert(sync_datas.end(), ret.begin(), ret.end());
    usleep(100);
    if (sync_datas.size() == state->availableDrones().size() - 1) {
      break;
    }
  }
  for (auto data : sync_datas) {
    updateWithDistributedVinsData(data);
  }
  if (params->verbose) {
    printf(
        "[ConsensusSolver::waitForSync@%d] receive finsish %ld/%ld time "
        "%.1f/%.1fms\n",
        self_id, sync_datas.size() + 1, state->availableDrones().size(),
        tic.toc(), config.timout_wait_sync);
  }
}

}  // namespace D2VINS