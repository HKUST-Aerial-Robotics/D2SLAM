#pragma once
#include <d2common/solver/ARock.hpp>
#include <d2common/d2pgo_types.h>

using namespace D2Common;

namespace D2PGO {
class D2PGO;
class ARockPGO: public ARockSolver {
protected:
    virtual void receiveAll() override;
    virtual void broadcastData() override;
    virtual void setStateProperties() override;
    void processPGOData(const DPGOData & data);
    D2PGO * pgo = nullptr;
    std::recursive_mutex pgo_data_mutex;
    std::vector<DPGOData> pgo_data;
    bool perturb_mode = true;
public:
    void inputDPGOData(const DPGOData & data);
    ARockPGO(D2State * _state, D2PGO * _pgo, ARockSolverConfig _config):
            ARockSolver(_state, _config), pgo(_pgo) {
    }
};
}