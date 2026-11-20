/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/whisper/runtime/whisper_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace trtmc {

inline int32_t resolve_whisper_expected_mel_length(const WhisperConfig& cfg) {
    return cfg.mel_length > 0 ? cfg.mel_length : cfg.max_source_positions * 2;
}

inline int32_t count_whisper_stride2_stages(int32_t mel_full, int32_t enc_full) {
    int32_t ratio = enc_full > 0 ? mel_full / enc_full : 2;
    int32_t stages = 0;
    while (ratio > 1) {
        ratio /= 2;
        ++stages;
    }
    return stages;
}

inline int32_t apply_whisper_stride2_subsampling(int32_t length, int32_t stages) {
    int32_t subsampled = length;
    for (int32_t stage = 0; stage < stages; ++stage) {
        subsampled = (subsampled + 2 - 3) / 2 + 1;
    }
    return subsampled;
}

inline int32_t compute_whisper_actual_encoder_length(int32_t mel_length, int32_t mel_full,
                                                     int32_t enc_full) {
    if (mel_full <= 0 || mel_length >= mel_full) {
        return 0;
    }

    const int32_t stages = count_whisper_stride2_stages(mel_full, enc_full);
    return apply_whisper_stride2_subsampling(mel_length, stages);
}

inline std::vector<float> build_whisper_padded_mel_input(const float* mel_data, int32_t mel_bins,
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

inline std::vector<float> build_whisper_encoder_mask_values(int32_t enc_seq, int32_t actual_enc) {
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

// The encoder input only an engine that projects the decoder's cross-attention
// K/V has ([positions, 1], FP32).
inline constexpr const char* kWhisperEncoderPaddedRows = "encoder_padded_rows";

// Values of the encoder_padded_rows input of an encoder engine that projects
// the decoder's cross-attention K/V: 1 for the encoder positions past the
// request's audio, 0 for every other position. The engine reads a row marked 1
// as a zero state (what zeroing the encoder output did before the projections
// moved into the engine) and leaves the others unchanged, so all zeros change
// nothing. actual_enc <= 0 means the audio fills the window.
inline std::vector<float> build_whisper_encoder_padded_rows(int32_t enc_seq, int32_t actual_enc) {
    if (enc_seq <= 0) {
        return {};
    }

    std::vector<float> padded(static_cast<std::size_t>(enc_seq), 0.0F);
    if (actual_enc <= 0 || actual_enc >= enc_seq) {
        return padded;
    }
    std::fill(padded.begin() + actual_enc, padded.end(), 1.0F);
    return padded;
}

inline std::vector<int32_t> make_whisper_initial_decoder_tokens(const WhisperConfig& cfg) {
    if (!cfg.decoder_start_token_ids.empty()) {
        return cfg.decoder_start_token_ids;
    }

    return {
        cfg.decoder_start_token_id,
        cfg.language_token_id,
        cfg.transcribe_token_id,
        cfg.notimestamps_token_id,
    };
}

} // namespace trtmc
