#pragma once
#include <swarm_msgs/relative_measurments.hpp>
#include "../d2pgo_config.h"
    
namespace D2Common {
    struct ARockSolverConfig;
    class DPGOData;
}

namespace D2PGO {
class D2PGO;
class PGOState;
class RotInit {
    void * rot_int = nullptr;
    bool enable_float32 = false;
    bool enable_arock = false;
public:
    RotInit(PGOState * _state, RotInitConfig _config, ARockSolverConfig arock_config, 
        bool enable_consenus, std::function<void(const DPGOData &)> _broadcastDataCallback);
    void addLoops(const std::vector<Swarm::LoopEdge> & good_loops);
    void setFixedFrameId(FrameIdType _fixed_frame_id);
    void inputDPGOData(const DPGOData & data);
    SolverReport solve(bool solve_6d=false);
    void reset();
};
}