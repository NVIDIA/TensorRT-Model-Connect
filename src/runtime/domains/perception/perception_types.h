#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct SegmentationConfig {
    int32_t num_classes{150};
    int32_t input_image_h{512};
    int32_t input_image_w{512};
    int32_t output_h{128};
    int32_t output_w{128};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

struct SegmentationResult {
    std::vector<int32_t> class_map; // [H, W] class indices
    int32_t height{0};
    int32_t width{0};
    int32_t num_classes{0};
};

} // namespace trtmc
