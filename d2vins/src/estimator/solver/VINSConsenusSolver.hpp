#pragma once
#include <d2common/solver/ConsensusSolver.hpp>
#include "ConsensusSync.hpp"

using namespace D2Common;
namespace D2VINS {

class D2Estimator;
class D2EstimatorState;

class D2VINSConsensusSolver : public ConsensusSolver {
protected:
    D2Estimator * estimator;
    SyncDataReceiver * receiver;
    void updateWithDistributedVinsData(const DistributedVinsData & dist_data);
    virtual void broadcastData() override;
    virtual void receiveAll() override;
    virtual void setStateProperties() override;
    virtual void waitForSync() override;
public:
    D2VINSConsensusSolver(D2Estimator * _estimator, D2EstimatorState * _state, SyncDataReceiver * _receiver, 
        ConsensusSolverConfig _config, int _solver_token): 
            estimator(_estimator), 
            ConsensusSolver((D2State*)_state, _config, _solver_token), receiver(_receiver) {
    }
};
}