#include <d2vins/d2vins_params.hpp>

namespace D2VINS {
D2VINSConfig * params = nullptr;

void initParams(ros::NodeHandle & nh) {
    params = new D2VINSConfig;
}
}