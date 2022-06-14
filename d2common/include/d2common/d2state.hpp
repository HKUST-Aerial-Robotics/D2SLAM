#pragma once
#include <set>

namespace D2Common {
class D2State {
protected:
    int self_id;
    std::set<int> all_drones;
public:
    D2State(int _self_id) :
        self_id(_self_id) {
    }
    std::set<int> availableDrones() const {
        return all_drones;
    }

    bool hasDrone(int drone_id) const{
        return all_drones.find(drone_id) != all_drones.end();
    }



};
};