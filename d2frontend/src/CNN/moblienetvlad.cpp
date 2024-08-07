#include "d2frontend/CNN/mobilenetvlad.h"

#include "d2common/utils.hpp"

#include <NvInfer.h>
#include <iostream>
#include <spdlog/spdlog.h>

#include "tensorrt_utils/buffers.h"
#include "tensorrt_utils/common.h"
#include "tensorrt_utils/logger.h"


namespace D2FrontEnd {
const int32_t knetvlad_desc_ra  w_size = 4096;

MobileNetVLAD::MobileNetVLAD(const MobileNetVLADConfig &netvald_config)
    : netvald_config_(netvald_config) {
    spdlog::info("MobileNetVLAD::MobileNetVLAD");
}

bool MobileNetVLAD::build() {
    spdlog::info("MobileNetVLAD::build");
    return true;
}

bool MobileNetVLAD::infer(const cv::Mat &image, std::vector<float> &descriptor) {
    spdlog::info("MobileNetVLAD::infer");
    return true;
}

void MobileNetVLAD::saveEngine() {
    spdlog::info("MobileNetVLAD::saveEngine");
}

bool MobileNetVLAD::deserializeEngine() {
    spdlog::info("MobileNetVLAD::deserializeEngine");
    return true;
}

}// namespace D2FrontEnd