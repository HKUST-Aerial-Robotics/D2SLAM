#pragma once
#include <d2common/solver/ARock.hpp>
#include "ConsensusSync.hpp"

using namespace D2Common;
namespace D2VINS {

class D2Estimator;
class D2EstimatorState;
class ARockVINS: public ARockSolver {
protected:
    D2Estimator * estimator;
    virtual void receiveAll() override;
    virtual void broadcastData() override;
    virtual void setStateProperties() override;
    void processDistributedVins(const DistributedVinsData & data);
    std::vector<DistributedVinsData> pgo_data;
public:
    void inputDPGOData(const DPGOData & data);
    ARockPGO(D2Estimator * _estimator, D2EstimatorState * _state, ARockSolverConfig _config):
        ARockSolver((D2State*)_state, _config, _solver_token),
        estimator(_estimator) {
    }
};
}