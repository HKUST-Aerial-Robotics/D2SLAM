#pragma once
#include "ARockPGO.hpp"

namespace D2PGO {
struct D2PGOConfig {
    int self_id;
    D2Common::ARockSolverConfig arock_config;
};

class D2PGO {
protected:
    D2PGOConfig config;
    int self_id;
public:
    D2PGO(D2PGOConfig _config):
        config(_config), self_id(_config.self_id) {
    }
};
}