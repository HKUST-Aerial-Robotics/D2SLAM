#pragma once

#include "d2frontend_params.h"
#include "loop_cam.h"

namespace D2Frontend {
    struct D2FTConfig {
        int foo;
    };

    class D2FeatureTracker {
        D2FTConfig _config;
        VisualImageDescArray * current_keyframe = nullptr;
    public:
        D2FeatureTracker(D2FTConfig config):
            _config(config)
        {
        }

        bool track(VisualImageDescArray * frames);
    };
} 