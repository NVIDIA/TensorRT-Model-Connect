/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/canary/runtime/canary_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace trtmc {

inline int32_t resolve_canary_expected_mel_length(const CanaryConfig& cfg) {
    return cfg.mel_length > 0 ? cfg.mel_length : cfg.max_source_positions * 2;
}

inline int32_t count_canary_stride2_stages(int32_t mel_full, int32_t enc_full) {
    int32_t ratio = enc_full > 0 ? mel_full / enc_full : 2;
    int32_t stages = 0;
    while (ratio > 1) {
        ratio /= 2;
        ++stages;
    }
    return stages;
}

inline int32_t apply_canary_stride2_subsampling(int32_t length, int32_t stages) {
    int32_t subsampled = length;
    for (int32_t stage = 0; stage < stages; ++stage) {
        subsampled = (subsampled + 2 - 3) / 2 + 1;
    }
    return subsampled;
}

inline int32_t compute_canary_actual_encoder_length(int32_t mel_length, int32_t mel_full,
                                                    int32_t enc_full) {
    if (mel_full <= 0 || mel_length >= mel_full) {
        return 0;
    }

    const int32_t stages = count_canary_stride2_stages(mel_full, enc_full);
    return apply_canary_stride2_subsampling(mel_length, stages);
}

inline std::vector<float> build_canary_padded_mel_input(const float* mel_data, int32_t mel_bins,
                                                        int32_t mel_length,
                                                        int32_t expected_length) {
    if (mel_data == nullptr || mel_bins <= 0 || mel_length <= 0 || expected_length <= 0) {
        return {};
    }

    std::vector<float> padded(
        static_cast<std::size_t>(mel_bins) * static_cast<std::size_t>(expected_length), 0.0F);
    const int32_t copy_len = std::min(mel_length, expected_length);
    for (int32_t bin = 0; bin < mel_bins; ++bin) {
        std::memcpy(padded.data() +
                        static_cast<std::size_t>(bin) * static_cast<std::size_t>(expected_length),
                    mel_data + static_cast<std::size_t>(bin) * static_cast<std::size_t>(mel_length),
                    static_cast<std::size_t>(copy_len) * sizeof(float));
    }
    return padded;
}

inline std::vector<float> build_canary_encoder_mask_values(int32_t enc_seq, int32_t actual_enc) {
    if (enc_seq <= 0) {
        return {};
    }

    std::vector<float> mask(static_cast<std::size_t>(enc_seq), 0.0F);
    const int32_t first_masked = std::clamp(actual_enc, 0, enc_seq);
    for (int32_t position = first_masked; position < enc_seq; ++position) {
        mask[static_cast<std::size_t>(position)] = -10000.0F;
    }
    return mask;
}

inline void quantize_canary_pcm16_inplace(std::vector<float>& samples) {
    for (float& sample : samples) {
        const float scaled = std::clamp(sample * 32768.0F, -32768.0F, 32767.0F);
        const auto pcm = static_cast<int16_t>(scaled);
        sample = static_cast<float>(pcm) / 32768.0F;
    }
}

inline std::vector<int32_t> make_canary_initial_decoder_tokens(const CanaryConfig& cfg) {
    return cfg.decoder_start_token_ids;
}

} // namespace trtmc
