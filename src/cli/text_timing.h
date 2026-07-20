/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace trtmc::cli {

struct TextTimingSample {
    double prefill_ms{0.0};
    double decode_ms{0.0};
    std::size_t generated_tokens{0};
};

struct TextBenchmarkMetrics {
    bool timing_available{false};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    double tokens_per_sec{0.0};
};

inline bool text_timing_available(double prefill_ms, double decode_ms) noexcept {
    return std::isfinite(prefill_ms) && prefill_ms >= 0.0 && std::isfinite(decode_ms) &&
           decode_ms > 0.0;
}

inline TextBenchmarkMetrics
summarize_text_benchmark(const std::vector<TextTimingSample>& samples) noexcept {
    if (samples.empty())
        return {};

    double prefill_total_ms = 0.0;
    double decode_total_ms = 0.0;
    std::size_t generated_tokens = 0;
    for (const auto& sample : samples) {
        if (!text_timing_available(sample.prefill_ms, sample.decode_ms))
            return {};
        prefill_total_ms += sample.prefill_ms;
        decode_total_ms += sample.decode_ms;
        generated_tokens += sample.generated_tokens;
    }

    if (!std::isfinite(prefill_total_ms) || !std::isfinite(decode_total_ms) ||
        decode_total_ms <= 0.0)
        return {};

    const double sample_count = static_cast<double>(samples.size());
    const double tokens_per_sec =
        static_cast<double>(generated_tokens) / (decode_total_ms / 1000.0);
    if (!std::isfinite(tokens_per_sec))
        return {};

    return {true, prefill_total_ms / sample_count, decode_total_ms / sample_count, tokens_per_sec};
}

inline std::string format_text_timing(double prefill_ms, double decode_ms) {
    std::ostringstream line;
    line << "[trtmc.timing] ";
    if (!text_timing_available(prefill_ms, decode_ms)) {
        line << "prefill_ms=unavailable decode_ms=unavailable total_ms=unavailable";
        return line.str();
    }
    line << std::fixed << std::setprecision(6) << "prefill_ms=" << prefill_ms
         << " decode_ms=" << decode_ms << " total_ms=" << (prefill_ms + decode_ms);
    return line.str();
}

inline std::string format_text_benchmark(const TextBenchmarkMetrics& metrics) {
    std::ostringstream line;
    line << "[trtmc.benchmark] ";
    if (!metrics.timing_available) {
        line << "prefill_ms=unavailable decode_ms=unavailable tokens_per_sec=unavailable";
        return line.str();
    }
    line << std::fixed << std::setprecision(2) << "prefill_ms=" << metrics.prefill_ms
         << " decode_ms=" << metrics.decode_ms << " tokens_per_sec=" << metrics.tokens_per_sec;
    return line.str();
}

} // namespace trtmc::cli
