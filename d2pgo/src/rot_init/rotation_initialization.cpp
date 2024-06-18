#include "rotation_initialization.hpp"

#include "rotation_initialization_arock.hpp"
#include "rotation_initialization_base.hpp"

namespace D2PGO {
RotInit::RotInit(PGOState *_state, RotInitConfig _config,
                 ARockSolverConfig arock_config, bool _enable_arock,
                 std::function<void(const DPGOData &)> _broadcastDataCallback)
    : enable_float32(_config.enable_float32), enable_arock(_enable_arock) {
  if (enable_arock) {
    if (enable_float32) {
      rot_int = new RotationInitARockf(_state, _config, arock_config,
                                       _broadcastDataCallback);
      printf("[D2PGO::RotInit] Mode: ARock + float32\n");
    } else {
      rot_int = new RotationInitARockd(_state, _config, arock_config,
                                       _broadcastDataCallback);
      printf("[D2PGO::RotInit] Mode: ARock + double\n");
    }
  } else {
    if (enable_float32) {
      rot_int = new RotationInitializationf(_state, _config);
      printf("[D2PGO::RotInit] Mode: float32\n");
    } else {
      rot_int = new RotationInitializationd(_state, _config);
      printf("[D2PGO::RotInit] Mode: double\n");
    }
  }
}

void RotInit::addLoops(const std::vector<Swarm::LoopEdge> &good_loops) {
  if (enable_arock) {
    if (enable_float32) {
      static_cast<RotationInitARockf *>(rot_int)->addLoops(good_loops);
    } else {
      static_cast<RotationInitARockd *>(rot_int)->addLoops(good_loops);
    }
  } else {
    if (enable_float32) {
      static_cast<RotationInitializationf *>(rot_int)->addLoops(good_loops);
    } else {
      static_cast<RotationInitializationd *>(rot_int)->addLoops(good_loops);
    }
  }
}

void RotInit::inputDPGOData(const DPGOData &data) {
  if (enable_arock) {
    if (enable_float32) {
      static_cast<RotationInitARockf *>(rot_int)->inputDPGOData(data);
    } else {
      static_cast<RotationInitARockd *>(rot_int)->inputDPGOData(data);
    }
  }
}

void RotInit::setFixedFrameId(FrameIdType _fixed_frame_id) {
  if (enable_arock) {
    if (enable_float32) {
      static_cast<RotationInitARockf *>(rot_int)->setFixedFrameId(
          _fixed_frame_id);
    } else {
      static_cast<RotationInitARockd *>(rot_int)->setFixedFrameId(
          _fixed_frame_id);
    }
  } else {
    if (enable_float32) {
      static_cast<RotationInitializationf *>(rot_int)->setFixedFrameId(
          _fixed_frame_id);
    } else {
      static_cast<RotationInitializationd *>(rot_int)->setFixedFrameId(
          _fixed_frame_id);
    }
  }
}

SolverReport RotInit::solve(bool solve_6d) {
  if (enable_arock) {
    if (enable_float32) {
      static_cast<RotationInitARockf *>(rot_int)->solve_6d = solve_6d;
      return static_cast<RotationInitARockf *>(rot_int)->solve();
    } else {
      static_cast<RotationInitARockd *>(rot_int)->solve_6d = solve_6d;
      return static_cast<RotationInitARockd *>(rot_int)->solve();
    }
  } else {
    if (enable_float32) {
      return static_cast<RotationInitializationf *>(rot_int)->solve();
    } else {
      return static_cast<RotationInitializationd *>(rot_int)->solve();
    }
  }
}

void RotInit::reset() {
  if (enable_arock) {
    if (enable_float32) {
      static_cast<RotationInitARockf *>(rot_int)->reset();
    } else {
      static_cast<RotationInitARockd *>(rot_int)->reset();
    }
  } else {
    if (enable_float32) {
      static_cast<RotationInitializationf *>(rot_int)->reset();
    } else {
      static_cast<RotationInitializationd *>(rot_int)->reset();
    }
  }
}
}  // namespace D2PGO