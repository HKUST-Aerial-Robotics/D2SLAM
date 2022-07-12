#include "ARockPGO.hpp"
#include "d2pgo.h"

namespace D2PGO {
void ARockPGO::receiveAll() {

}

void ARockPGO::broadcastData() {

}

void ARockPGO::setStateProperties() {
    pgo->setStateProperties(getProblem());
}

};