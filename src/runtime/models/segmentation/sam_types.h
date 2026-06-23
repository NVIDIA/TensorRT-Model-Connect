#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct SamConfig {
    int32_t image_size{1024};
    int32_t image_embedding_size{64}; // image_size / patch_size
    int32_t decoder_hidden_size{256};
    int32_t num_mask_outputs{4}; // num_multimask + 1
    int32_t num_multimask_outputs{3};
    std::vector<float> image_mean{0.485F, 0.456F, 0.406F};
    std::vector<float> image_std{0.229F, 0.224F, 0.225F};

    // Prompt encoder embeddings loaded from config.json.
    std::vector<float> point_embed_fg;    // foreground point [decoder_hidden_size]
    std::vector<float> point_embed_bg;    // background point [decoder_hidden_size]
    std::vector<float> not_a_point_embed; // padding point [decoder_hidden_size]
    std::vector<float> shared_image_pe;   // [2, num_pos_feats] flattened
};

struct SamResult {
    std::vector<float> masks;      // [num_masks, 256, 256]
    std::vector<float> iou_scores; // [num_masks]
    int32_t num_masks{0};
    int32_t mask_height{256};
    int32_t mask_width{256};
};

} // namespace trtmc
