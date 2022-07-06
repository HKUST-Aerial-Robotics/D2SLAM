#pragma once
#include <d2common/solver/ARock.hpp>

using namespace D2Common;

namespace D2PGO {
class ARockPGO: public ARockSolver {
protected:
    virtual void receiveAll() override;
    virtual void broadcastData() override;
    virtual void setStateProperties() override;
public:
    ARockPGO(D2State * _state, ARockSolverConfig _config):
            ARockSolver(_state, _config) {
    }
};
}