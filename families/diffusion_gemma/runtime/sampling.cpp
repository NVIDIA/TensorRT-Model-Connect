/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/diffusion_gemma/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace trtmc::diffusion_gemma {

std::vector<float> attention_mask(std::int64_t sequence_length, std::int64_t past_length,
                                  std::int64_t sliding_window, bool decoder) {
    if (sequence_length < 1 || past_length < 0 || sliding_window < 0)
        throw std::invalid_argument("diffusion_gemma attention dimensions are invalid");
    const auto total = sequence_length + past_length;
    std::vector<float> result(static_cast<std::size_t>(sequence_length * total));
    for (std::int64_t row = 0; row < sequence_length; ++row) {
        const auto position = past_length + row;
        for (std::int64_t column = 0; column < total; ++column) {
            bool allowed = decoder || column <= position;
            if (sliding_window != 0 && !decoder) {
                allowed = allowed && column > position - sliding_window;
            }
            result[static_cast<std::size_t>(row * total + column)] =
                allowed ? 0.0F : -std::numeric_limits<float>::infinity();
        }
    }
    return result;
}

CanvasSample sample_canvas(const float* raw_logits, std::size_t canvas_length,
                           std::size_t vocabulary_size, float temperature, double entropy_bound,
                           std::mt19937_64& random) {
    if (raw_logits == nullptr || canvas_length == 0 || vocabulary_size == 0 ||
        !(temperature > 0.0F) || !std::isfinite(temperature) || !(entropy_bound >= 0.0))
        throw std::invalid_argument("diffusion_gemma sampler inputs are invalid");
    CanvasSample result;
    result.argmax.resize(canvas_length);
    result.next_canvas.resize(canvas_length);
    std::vector<double> entropies(canvas_length);
    std::vector<std::size_t> order(canvas_length);
    std::vector<double> masses(vocabulary_size);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (std::size_t row = 0; row < canvas_length; ++row) {
        const float* logits = raw_logits + row * vocabulary_size;
        const float maximum = *std::max_element(logits, logits + vocabulary_size);
        if (!std::isfinite(maximum))
            throw std::runtime_error("diffusion_gemma returned non-finite logits");
        const auto argmax =
            static_cast<std::size_t>(std::max_element(logits, logits + vocabulary_size) - logits);
        result.argmax[row] = static_cast<std::int32_t>(argmax);
        double total_mass = 0.0;
        for (std::size_t token = 0; token < vocabulary_size; ++token) {
            const float logit = logits[token];
            if (!std::isfinite(logit))
                throw std::runtime_error("diffusion_gemma returned non-finite logits");
            masses[token] = std::exp(static_cast<double>((logit - maximum) / temperature));
            total_mass += masses[token];
        }
        const double threshold = uniform(random) * total_mass;
        double cumulative = 0.0;
        double entropy = 0.0;
        std::size_t selected = vocabulary_size - 1;
        bool sampled = false;
        for (std::size_t token = 0; token < vocabulary_size; ++token) {
            cumulative += masses[token];
            if (!sampled && cumulative > threshold) {
                selected = token;
                sampled = true;
            }
            const double probability = masses[token] / total_mass;
            if (probability > 0.0)
                entropy -= probability * std::log(probability);
        }
        result.next_canvas[row] = static_cast<std::int32_t>(selected);
        entropies[row] = entropy;
        result.mean_entropy += entropy / static_cast<double>(canvas_length);
    }
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return entropies[left] < entropies[right];
    });
    double preceding_entropy = 0.0;
    std::uniform_int_distribution<std::int32_t> token(
        0, static_cast<std::int32_t>(vocabulary_size - 1));
    for (const auto index : order) {
        // Acceptance is recomputed from this step's complete entropy ordering.
        // It is not a monotone set of permanently accepted tokens.
        if (preceding_entropy > entropy_bound)
            result.next_canvas[index] = token(random);
        preceding_entropy += entropies[index];
    }
    return result;
}

} // namespace trtmc::diffusion_gemma
