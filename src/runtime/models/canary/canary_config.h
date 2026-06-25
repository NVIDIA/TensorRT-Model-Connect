#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct CanaryConfig {
    int32_t num_mel_bins{80};
    int32_t max_source_positions{1500};
    int32_t max_target_positions{448};
    int32_t encoder_layers{0};
    int32_t decoder_layers{0};
    int32_t decoder_attention_heads{0};
    int32_t decoder_start_token_id{50258}; // <|startoftranscript|>
    int32_t language_token_id{50259};      // <|en|>
    int32_t translate_token_id{50358};
    int32_t transcribe_token_id{50359};
    int32_t notimestamps_token_id{50363};
    int32_t eot_token_id{50257}; // <|endoftext|>
    std::string language{"en"};

    int32_t mel_length{0}; // expected mel input length; 0 = auto (max_source_positions * 2)

    // Custom decoder start sequence (overrides individual token fields above).
    // When non-empty, transcribe() uses this instead of building from individual fields.
    // Set from bundle config.json "decoder_start_token_ids" for compatible ASR variants.
    std::vector<int32_t> decoder_start_token_ids;
};

} // namespace trtmc
