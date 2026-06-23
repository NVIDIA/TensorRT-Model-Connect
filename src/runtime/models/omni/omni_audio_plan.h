#pragma once

#include "runtime/models/omni/omni_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace trtmc {

struct OmniAudioEncodePlan {
    int32_t actual_frames{0};
    int32_t output_frames{0};
    int32_t embed_dim{0};
    std::size_t input_size{0};
    std::size_t copy_size{0};
    std::size_t output_elements{0};
};

struct OmniTalkerPlan {
    bool should_run_talker{false};
    int32_t num_tokens{0};
};

struct OmniCodecPlan {
    bool should_run_codec{false};
    int32_t n_codebooks{0};
    int32_t n_frames{0};
};

struct OmniTalkerDecodePlan {
    int32_t n_codebooks{0};
    int32_t codebook_size{0};
    int32_t num_tokens{0};
};

inline OmniAudioEncodePlan make_omni_audio_encode_plan(const OmniConfig& config,
                                                       int32_t num_mel_bins, int32_t num_frames) {
    OmniAudioEncodePlan plan;
    plan.actual_frames = std::min(num_frames, config.audio_num_frames);
    plan.output_frames = plan.actual_frames / 2;
    plan.embed_dim = config.audio_embed_dim;
    plan.input_size =
        static_cast<std::size_t>(num_mel_bins) * static_cast<std::size_t>(config.audio_num_frames);
    plan.copy_size =
        static_cast<std::size_t>(num_mel_bins) * static_cast<std::size_t>(plan.actual_frames);
    plan.output_elements =
        static_cast<std::size_t>(plan.output_frames) * static_cast<std::size_t>(plan.embed_dim);
    return plan;
}

inline std::vector<float> build_omni_audio_encoder_input(const float* mel_features,
                                                         const OmniAudioEncodePlan& plan) {
    std::vector<float> input_padded(plan.input_size, 0.0F);
    if (mel_features != nullptr && plan.copy_size > 0) {
        std::memcpy(input_padded.data(), mel_features, plan.copy_size * sizeof(float));
    }
    return input_padded;
}

inline OmniTalkerPlan make_omni_talker_plan(std::size_t text_token_count,
                                            std::size_t hidden_state_count,
                                            bool has_talker_engine) {
    OmniTalkerPlan plan;
    plan.should_run_talker = has_talker_engine && text_token_count > 0 && hidden_state_count > 0;
    plan.num_tokens = static_cast<int32_t>(text_token_count);
    return plan;
}

inline OmniCodecPlan make_omni_codec_plan(const OmniConfig& config, std::size_t codec_token_count) {
    OmniCodecPlan plan;
    plan.n_codebooks = config.talker_n_codebooks;
    plan.should_run_codec = plan.n_codebooks > 0 && codec_token_count > 0;
    plan.n_frames =
        plan.should_run_codec ? static_cast<int32_t>(codec_token_count) / plan.n_codebooks : 0;
    plan.should_run_codec = plan.should_run_codec && plan.n_frames > 0;
    return plan;
}

inline OmniTalkerDecodePlan make_omni_talker_decode_plan(int32_t n_codebooks, int32_t codebook_size,
                                                         int32_t num_tokens) {
    OmniTalkerDecodePlan plan;
    plan.n_codebooks = n_codebooks;
    plan.codebook_size = codebook_size;
    plan.num_tokens = num_tokens;
    return plan;
}

inline int32_t select_omni_codebook_argmax(const std::vector<float>& logits, int32_t offset,
                                           int32_t codebook_size) {
    if (offset + codebook_size > static_cast<int32_t>(logits.size())) {
        return 0;
    }

    int32_t best = 0;
    for (int32_t index = 1; index < codebook_size; ++index) {
        if (logits[offset + index] > logits[offset + best]) {
            best = index;
        }
    }
    return best;
}

inline void append_omni_talker_codes_from_logits(const std::vector<float>& logits,
                                                 const OmniTalkerDecodePlan& plan,
                                                 std::vector<int32_t>& all_codes) {
    for (int32_t codebook = 0; codebook < plan.n_codebooks; ++codebook) {
        const int32_t offset = codebook * plan.codebook_size;
        all_codes.push_back(select_omni_codebook_argmax(logits, offset, plan.codebook_size));
    }
}

inline std::vector<int32_t>
build_omni_code2wav_input_codes(const std::vector<int32_t>& codec_tokens, int32_t n_codebooks,
                                int32_t max_frames, int32_t actual_frames) {
    const auto input_size = static_cast<std::size_t>(n_codebooks) * max_frames;
    std::vector<int32_t> input_codes(input_size, 0);
    for (int32_t codebook = 0; codebook < n_codebooks; ++codebook) {
        for (int32_t frame = 0; frame < actual_frames; ++frame) {
            input_codes[static_cast<std::size_t>(codebook) * max_frames + frame] =
                codec_tokens[static_cast<std::size_t>(frame) * n_codebooks + codebook];
        }
    }
    return input_codes;
}

} // namespace trtmc
