#pragma once
#include "d2common/d2landmarks.h"

using namespace D2Common;

namespace D2FrontEnd {
class LandmarkManager {
protected:
    std::map<LandmarkIdType, LandmarkPerId> landmark_db;
    int count = 0;
public:
    virtual int addLandmark(const LandmarkPerFrame & lm);
    virtual void updateLandmark(const LandmarkPerFrame & lm);
    bool hasLandmark(const LandmarkIdType & id) const {
        return landmark_db.find(id) != landmark_db.end();
    }
    LandmarkPerId & at(int i) {
        return landmark_db.at(i);
    }
};
}