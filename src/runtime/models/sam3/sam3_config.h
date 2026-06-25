#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct Sam3Config {
    int32_t text_max_position_embeddings{32};
    int32_t text_pad_token_id{1};
    int32_t text_projection_dim{512};
    int32_t image_size{1008};
    int32_t low_res_mask_size{288};
    int32_t num_queries{200};
    float score_threshold{0.5F};
    float mask_threshold{0.5F};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};
};

} // namespace trtmc
