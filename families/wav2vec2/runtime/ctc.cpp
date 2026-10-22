/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/wav2vec2/runtime/ctc.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace trtmc::wav2vec2 {

namespace {
// The reference divides by sqrt(var + 1e-7); the epsilon is inside the root.
constexpr float kNormalizeEpsilon = 1e-7F;
} // namespace

std::vector<float> mix_mono(const float* samples, std::int64_t count, std::int32_t channels) {
    if (samples == nullptr || count <= 0)
        throw std::invalid_argument("wav2vec2 requires float32 PCM input");
    if (channels <= 0)
        throw std::invalid_argument("wav2vec2 requires at least one audio channel");
    if (count % channels != 0)
        throw std::invalid_argument("wav2vec2 input is not a whole number of audio frames");

    const std::int64_t frames = count / channels;
    std::vector<float> mono(static_cast<std::size_t>(frames));
    const auto scale = 1.0F / static_cast<float>(channels);
    for (std::int64_t frame = 0; frame < frames; ++frame) {
        float total = 0.0F;
        for (std::int32_t channel = 0; channel < channels; ++channel)
            total += samples[frame * channels + channel];
        mono[static_cast<std::size_t>(frame)] = total * scale;
    }
    return mono;
}

void normalize_waveform(std::vector<float>& samples) {
    if (samples.empty())
        throw std::invalid_argument("wav2vec2 cannot normalise an empty clip");

    double total = 0.0;
    for (const auto value : samples)
        total += static_cast<double>(value);
    const double mean = total / static_cast<double>(samples.size());

    double squared = 0.0;
    for (const auto value : samples) {
        const double centred = static_cast<double>(value) - mean;
        squared += centred * centred;
    }
    const double variance = squared / static_cast<double>(samples.size());
    const auto deviation = std::sqrt(variance + static_cast<double>(kNormalizeEpsilon));
    for (auto& value : samples)
        value = static_cast<float>((static_cast<double>(value) - mean) / deviation);
}

std::int64_t feature_frames(std::int64_t samples, const std::vector<std::int32_t>& kernels,
                            const std::vector<std::int32_t>& strides) {
    if (kernels.size() != strides.size() || kernels.empty())
        throw std::invalid_argument("wav2vec2 needs one kernel and stride per encoder layer");
    std::int64_t length = samples;
    for (std::size_t layer = 0; layer < kernels.size(); ++layer) {
        const auto kernel = static_cast<std::int64_t>(kernels[layer]);
        const auto stride = static_cast<std::int64_t>(strides[layer]);
        if (kernel <= 0 || stride <= 0)
            throw std::invalid_argument("wav2vec2 encoder geometry is not positive");
        if (length < kernel)
            return 0;
        length = (length - kernel) / stride + 1;
    }
    return length;
}

CtcDecoded greedy_decode(const float* logits, std::int64_t frames, std::int64_t vocab_size,
                         const CtcVocabulary& vocabulary) {
    if (logits == nullptr || frames <= 0 || vocab_size <= 0)
        throw std::invalid_argument("wav2vec2 cannot decode an empty logit block");
    if (static_cast<std::int64_t>(vocabulary.tokens.size()) != vocab_size)
        throw std::invalid_argument("wav2vec2 vocabulary does not match the logit width");

    CtcDecoded decoded;
    std::int32_t previous = -1;
    for (std::int64_t frame = 0; frame < frames; ++frame) {
        const float* row = logits + frame * vocab_size;
        std::int64_t best = 0;
        for (std::int64_t index = 1; index < vocab_size; ++index)
            if (row[index] > row[best])
                best = index;
        if (!std::isfinite(row[best]))
            throw std::runtime_error("wav2vec2 engine returned a non-finite logit");

        const auto token_id = static_cast<std::int32_t>(best);
        // Collapse first, then drop the blank. A blank between two identical
        // frames is exactly what keeps a doubled letter from being collapsed.
        if (token_id == previous) {
            previous = token_id;
            continue;
        }
        previous = token_id;
        if (token_id == vocabulary.blank_token_id)
            continue;
        decoded.token_ids.push_back(token_id);
        const auto& token = vocabulary.tokens[static_cast<std::size_t>(token_id)];
        decoded.text += (token == vocabulary.word_delimiter_token) ? std::string(" ") : token;
    }
    return decoded;
}

} // namespace trtmc::wav2vec2
