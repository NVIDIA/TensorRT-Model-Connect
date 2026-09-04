/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace trtmc {

inline float qwen3_omni_round_bfloat16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits &= 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

inline std::vector<std::int32_t>
qwen3_omni_code2wav_input(const std::vector<std::int32_t>& frame_major_codes,
                          std::int32_t num_codebooks, std::int32_t max_frames) {
    if (num_codebooks <= 0 || max_frames <= 0 ||
        frame_major_codes.size() % static_cast<std::size_t>(num_codebooks) != 0) {
        throw std::invalid_argument("invalid Qwen3-Omni codec layout");
    }
    const auto available_frames = static_cast<std::int32_t>(
        frame_major_codes.size() / static_cast<std::size_t>(num_codebooks));
    if (available_frames > max_frames)
        throw std::invalid_argument("Qwen3-Omni codec frames exceed Code2Wav capacity");
    const auto actual_frames = available_frames;
    std::vector<std::int32_t> result(
        static_cast<std::size_t>(num_codebooks) * static_cast<std::size_t>(max_frames), 0);
    for (std::int32_t codebook = 0; codebook < num_codebooks; ++codebook) {
        for (std::int32_t frame = 0; frame < actual_frames; ++frame) {
            result[static_cast<std::size_t>(codebook) * max_frames + frame] =
                frame_major_codes[static_cast<std::size_t>(frame) * num_codebooks + codebook];
        }
    }
    return result;
}

inline std::size_t qwen3_omni_output_samples(std::int32_t frames, std::int32_t upsample_factor,
                                             std::int32_t output_delay,
                                             std::size_t engine_samples) {
    if (frames <= 0 || upsample_factor <= 0)
        return 0;
    if (output_delay < 0)
        throw std::invalid_argument("Qwen3-Omni Code2Wav delay must be non-negative");
    const auto untrimmed = static_cast<std::size_t>(frames) * upsample_factor;
    const auto delay = static_cast<std::size_t>(output_delay);
    if (untrimmed <= delay)
        throw std::invalid_argument("Qwen3-Omni Code2Wav delay exceeds generated audio");
    const auto model_samples = untrimmed - delay;
    if (engine_samples < model_samples)
        throw std::runtime_error("Qwen3-Omni Code2Wav output is shorter than its contract");
    return model_samples;
}

inline bool qwen3_omni_embedding_section_size(std::size_t bytes, std::int32_t tables,
                                              std::int32_t vocab_size, std::int32_t hidden_size) {
    if (tables <= 0 || vocab_size <= 0 || hidden_size <= 0)
        return false;
    return bytes == static_cast<std::size_t>(tables) * vocab_size * hidden_size * sizeof(float);
}

inline std::int32_t
qwen3_omni_talker_argmax(const std::vector<float>& logits, std::int32_t codebook_size,
                         std::int32_t eos_token,
                         const std::vector<std::int32_t>& generated_coarse_codes) {
    if (codebook_size <= 0 || static_cast<std::size_t>(codebook_size) > logits.size() ||
        eos_token < 0 || static_cast<std::size_t>(eos_token) >= logits.size()) {
        throw std::invalid_argument("Qwen3-Omni Talker logits do not match the codec vocabulary");
    }
    constexpr float kRepetitionPenalty = 1.05F;
    const auto score = [&](std::int32_t token) {
        float value = logits[static_cast<std::size_t>(token)];
        if (std::find(generated_coarse_codes.begin(), generated_coarse_codes.end(), token) !=
            generated_coarse_codes.end()) {
            value = value < 0.0F ? value * kRepetitionPenalty : value / kRepetitionPenalty;
        }
        return value;
    };
    std::int32_t best = 0;
    float best_score = score(best);
    for (std::int32_t token = 1; token < codebook_size; ++token) {
        const float token_score = score(token);
        if (token_score > best_score) {
            best = token;
            best_score = token_score;
        }
    }
    if (const float eos_score = score(eos_token); eos_score > best_score)
        best = eos_token;
    return best;
}

inline std::int32_t qwen3_omni_audio_frame_limit(std::int32_t talker_max_new_tokens) {
    return std::max(talker_max_new_tokens - 1, 0);
}

} // namespace trtmc
