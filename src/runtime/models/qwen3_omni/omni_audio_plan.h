/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/qwen3_omni/omni_config.h"

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

struct OmniCodecPlan {
    bool should_run_codec{false};
    int32_t n_codebooks{0};
    int32_t n_frames{0};
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

inline OmniCodecPlan make_omni_codec_plan(const OmniConfig& config, std::size_t codec_token_count) {
    OmniCodecPlan plan;
    plan.n_codebooks = config.talker_n_codebooks;
    plan.should_run_codec = plan.n_codebooks > 0 && codec_token_count > 0;
    plan.n_frames =
        plan.should_run_codec ? static_cast<int32_t>(codec_token_count) / plan.n_codebooks : 0;
    plan.should_run_codec = plan.should_run_codec && plan.n_frames > 0;
    return plan;
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

inline std::size_t code2wav_output_samples(const OmniConfig& config, int32_t actual_frames,
                                           std::size_t engine_output_samples) {
    if (actual_frames <= 0 || config.code2wav_upsample_factor <= 0)
        return 0;
    const auto untrimmed = static_cast<std::size_t>(actual_frames) *
                           static_cast<std::size_t>(config.code2wav_upsample_factor);
    const auto delay = static_cast<std::size_t>(std::max(config.code2wav_output_delay, 0));
    const auto model_samples = untrimmed > delay ? untrimmed - delay : 0;
    return std::min(engine_output_samples, model_samples);
}

} // namespace trtmc
