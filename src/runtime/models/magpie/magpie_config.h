#pragma once

#include <cstdint>

namespace trtmc {

struct MagpieTTSConfig {
    int32_t sample_rate{22050};
    int32_t hidden_size{0};
    int32_t num_codebooks{8};
    int32_t codebook_size{2024};
    float frames_per_second{21.5F};
    int32_t num_speakers{5};
    int32_t encoder_layers{6};
    int32_t decoder_layers{12};
    int32_t text_vocab_size{0};
    int32_t max_source_positions{2048};
    int32_t xa_n_heads{1};
    int32_t xa_d_head{128};
    float temperature{0.6F};
    int32_t top_k{80};
    bool greedy{false};
    float cfg_scale{2.5F};
    int32_t finished_limit_with_eot{0};
    bool enable_finished_limit_stop{false};
    int64_t seed{-1};
};

} // namespace trtmc
