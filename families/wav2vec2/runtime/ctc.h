/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::wav2vec2 {

struct CtcVocabulary {
    std::vector<std::string> tokens;
    std::int32_t blank_token_id{0};
    std::string word_delimiter_token{"|"};
};

struct CtcDecoded {
    std::string text;
    std::vector<std::int32_t> token_ids;
};

// Average interleaved channels down to one. The reference is only ever fed
// mono, so a multi-channel clip is reduced before anything else happens.
std::vector<float> mix_mono(const float* samples, std::int64_t count, std::int32_t channels);

// The feature extractor's do_normalize rule: zero mean, unit variance over the
// whole clip. The variance is taken without Bessel's correction and floored by
// the same epsilon the reference uses, so a constant clip stays finite.
void normalize_waveform(std::vector<float>& samples);

// The output frame count for a sample count, applying each convolution's
// floor((n - kernel) / stride) + 1 in turn. Used to check the engine against
// the geometry the bundle declares rather than trusting the returned shape.
std::int64_t feature_frames(std::int64_t samples, const std::vector<std::int32_t>& kernels,
                            const std::vector<std::int32_t>& strides);

// Greedy CTC: take each frame's argmax, collapse runs of the same id, drop the
// blank, and map the word delimiter to a space. Repeats are collapsed before
// the blank is dropped, which is what lets a doubled letter survive when the
// reference emits a blank between its two frames.
CtcDecoded greedy_decode(const float* logits, std::int64_t frames, std::int64_t vocab_size,
                         const CtcVocabulary& vocabulary);

} // namespace trtmc::wav2vec2
