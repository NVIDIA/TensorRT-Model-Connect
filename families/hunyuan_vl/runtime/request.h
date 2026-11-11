/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::hunyuan_vl {
using namespace internal;

inline Span<const ConfigField> generation_fields() {
    static const std::vector<ConfigField> fields{
        {"max_new_tokens", ConfigKind::I64, int64_t{1024}, "Maximum generated tokens"},
        {"temperature", ConfigKind::F64, 0.0, "Sampling temperature; zero is greedy"},
        {"top_k", ConfigKind::I64, int64_t{0}, "Top-k filter when sampling; zero keeps all"},
        {"top_p", ConfigKind::F64, 1.0, "Nucleus filter when sampling"},
        {"repetition_penalty", ConfigKind::F64, 1.0,
         "Penalty on tokens already in the prompt or output"},
        {"seed", ConfigKind::I64, int64_t{-1}, "Sampling seed; -1 is a fixed default"},
    };
    return {fields.data(), fields.size()};
}

struct Generation {
    int32_t max_new_tokens{};
    double temperature{};
    int32_t top_k{};
    double top_p{};
    double repetition_penalty{};
    int64_t seed{};
};

inline Generation read_generation(ConfigView supplied) {
    const auto fields = generation_fields();
    validate_config(fields, supplied);
    Generation g;
    const auto bounded = [&](const char* name, int64_t low, int64_t high) {
        const auto value = *config_get<int64_t>(supplied, fields, name);
        if (value < low || value > high)
            throw ConfigError(std::string("HunYuanVL ") + name + " is out of range");
        return value;
    };
    g.max_new_tokens = int32_t(bounded("max_new_tokens", 1, std::numeric_limits<int32_t>::max()));
    g.top_k = int32_t(bounded("top_k", 0, std::numeric_limits<int32_t>::max()));
    g.seed = bounded("seed", -1, std::numeric_limits<int64_t>::max());
    g.temperature = *config_get<double>(supplied, fields, "temperature");
    g.top_p = *config_get<double>(supplied, fields, "top_p");
    g.repetition_penalty = *config_get<double>(supplied, fields, "repetition_penalty");
    if (!std::isfinite(g.temperature) || g.temperature < 0 || !std::isfinite(g.top_p) ||
        g.top_p <= 0 || g.top_p > 1 || !std::isfinite(g.repetition_penalty) ||
        g.repetition_penalty <= 0)
        throw ConfigError("HunYuanVL temperature, top_p or repetition_penalty is out of range");
    return g;
}

// Greedy (first maximum) at temperature zero; otherwise temperature, top-k and top-p sampling
// with a deterministic xorshift64 stream.
class Sampler {
  public:
    explicit Sampler(const Generation& g)
        : g_(g), state_(g.seed < 0              ? 42U
                        : uint64_t(g.seed) == 0 ? 1U
                                                : uint64_t(g.seed)) {}
    int32_t operator()(const std::vector<float>& logits) {
        const int32_t vocab = int32_t(logits.size());
        if (g_.temperature == 0)
            return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::vector<int32_t> order(static_cast<size_t>(vocab));
        std::iota(order.begin(), order.end(), 0);
        const int32_t k = g_.top_k <= 0 ? vocab : std::min(g_.top_k, vocab);
        std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) {
            return logits[size_t(a)] > logits[size_t(b)];
        });
        std::vector<double> probabilities(static_cast<size_t>(k));
        double total = 0;
        for (int32_t i = 0; i < k; ++i) {
            probabilities[size_t(i)] =
                std::exp((double(logits[size_t(order[size_t(i)])]) - logits[size_t(order[0])]) /
                         g_.temperature);
            total += probabilities[size_t(i)];
        }
        int32_t keep = k;
        if (g_.top_p < 1) {
            double cumulative = 0;
            for (keep = 0; keep < k;) {
                cumulative += probabilities[size_t(keep++)] / total;
                if (cumulative >= g_.top_p)
                    break;
            }
        }
        double kept = 0;
        for (int32_t i = 0; i < keep; ++i)
            kept += probabilities[size_t(i)];
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        const double u = double(state_ >> 11) / double(uint64_t{1} << 53) * kept;
        double cumulative = 0;
        for (int32_t i = 0; i < keep; ++i) {
            cumulative += probabilities[size_t(i)];
            if (u < cumulative)
                return order[size_t(i)];
        }
        return order[size_t(keep - 1)];
    }

  private:
    Generation g_;
    uint64_t state_;
};
} // namespace trtmc::hunyuan_vl
