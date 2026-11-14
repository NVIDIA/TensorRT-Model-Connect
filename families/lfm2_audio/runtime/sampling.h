/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace trtmc::lfm2_audio {
inline float bf16(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits = (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U;
    std::memcpy(&x, &bits, sizeof(bits));
    return x;
}

inline std::vector<float> probabilities(const std::vector<float>& logits, float temperature,
                                        int top_k) {
    if (logits.empty() || temperature <= 0 || !std::isfinite(temperature) || top_k < 0 ||
        top_k > static_cast<int>(logits.size()))
        throw std::invalid_argument("invalid LFM audio sampling configuration");
    auto scores = logits;
    for (auto& value : scores)
        value = bf16(value / temperature);
    float threshold = -INFINITY;
    if (top_k > 0) {
        auto sorted = scores;
        std::nth_element(sorted.begin(), sorted.begin() + top_k - 1, sorted.end(),
                         std::greater<float>());
        threshold = sorted[top_k - 1];
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    float total = 0;
    for (auto& value : scores) {
        value = value < threshold ? 0 : std::exp(value - maximum);
        total += value;
    }
    for (auto& value : scores)
        value = bf16(value / total);
    return scores;
}

// A shared uniform variate permits deterministic comparison with the reference
// distribution. Native mt19937 draws do not reproduce torch.multinomial seeds.
inline int sample_probabilities(const std::vector<float>& probs, double uniform) {
    if (probs.empty() || !std::isfinite(uniform) || uniform < 0 || uniform >= 1)
        throw std::invalid_argument("invalid LFM audio sampling uniform");
    double total = 0;
    for (const float value : probs) {
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("invalid LFM audio sampling probability");
        total += value;
    }
    if (total <= 0)
        throw std::invalid_argument("empty LFM audio sampling distribution");
    const double target = uniform * total;
    double cumulative = 0;
    int last_positive = 0;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (probs[i] > 0)
            last_positive = static_cast<int>(i);
        if (target < cumulative)
            return static_cast<int>(i);
    }
    return last_positive;
}

inline int sample(const std::vector<float>& logits, float temperature, int top_k,
                  std::mt19937& rng) {
    if (logits.empty())
        throw std::invalid_argument("empty LFM audio logits");
    if (temperature <= 0 || top_k == 1)
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    const auto probs = probabilities(logits, temperature, top_k);
    return sample_probabilities(probs, std::uniform_real_distribution<double>{0, 1}(rng));
}
} // namespace trtmc::lfm2_audio
