/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Wall-clock stage times of a Canary request, reported through TextResult:
//   setup_ms    audio frontend (resampling and mel features)
//   prefill_ms  encoder and cross-attention setup
//   decode_ms   decoder loop (prompt and generated steps)
// They only observe the request: no stage reads them.

#include "trtmc/task.h"

#include <chrono>
#include <cstddef>
#include <vector>

namespace trtmc {

using CanaryStageClock = std::chrono::steady_clock;

inline double canary_stage_elapsed_ms(CanaryStageClock::time_point started) {
    return std::chrono::duration<double, std::milli>(CanaryStageClock::now() - started).count();
}

struct CanaryStageTimes {
    double setup_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
};

inline void add_canary_stage_times(TextResult& combined, const TextResult& segment) {
    combined.setup_ms += segment.setup_ms;
    combined.prefill_ms += segment.prefill_ms;
    combined.decode_ms += segment.decode_ms;
}

// A batched chunk runs each stage once for all of its segments. Every segment
// the frontend prepared takes an equal share of the frontend wall, and every
// segment that reached the engines an equal share of the encoder and decoder
// walls, so summing a request's segments never counts a wall twice.
inline void share_canary_chunk_stage_times(const CanaryStageTimes& chunk,
                                           const std::vector<std::size_t>& indices,
                                           std::size_t chunk_start, std::size_t chunk_end,
                                           const std::vector<std::size_t>& valid_indices,
                                           std::vector<TextResult>& segment_results) {
    if (chunk_end > chunk_start) {
        const double setup_share = chunk.setup_ms / static_cast<double>(chunk_end - chunk_start);
        for (std::size_t cursor = chunk_start; cursor < chunk_end; ++cursor)
            segment_results[indices[cursor]].setup_ms = setup_share;
    }
    if (valid_indices.empty())
        return;
    const auto engine_segments = static_cast<double>(valid_indices.size());
    for (const std::size_t index : valid_indices) {
        segment_results[index].prefill_ms = chunk.prefill_ms / engine_segments;
        segment_results[index].decode_ms = chunk.decode_ms / engine_segments;
    }
}

} // namespace trtmc
