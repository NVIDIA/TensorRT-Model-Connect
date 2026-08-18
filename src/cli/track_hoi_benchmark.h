/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::cli {

struct TrackHoiLatencySummary {
    double mean_ms{0.0};
    double median_ms{0.0};
    double p90_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
};

// Python round() ties to even. The archived reference computes its p90 index
// as round((sample_count - 1) * 0.90), so keep that rule exact for every N.
std::size_t track_hoi_p90_index(std::size_t sample_count);

TrackHoiLatencySummary summarize_track_hoi_latency(const std::vector<double>& samples_ms);

void validate_track_hoi_output_paths(const std::string& output_json,
                                     const std::string& output_masks_dir,
                                     const std::string& benchmark_json,
                                     std::size_t input_frame_count);

std::string render_track_hoi_benchmark_receipt(
    const std::string& benchmark_scope, int warmup, std::size_t input_frame_count,
    int32_t produced_frame_count, const std::vector<double>& samples_ms,
    const TrackHoiLatencySummary& summary, const std::string& frame_decode_mode,
    std::size_t frame_decode_max_concurrency, const std::string& output_json,
    const std::string& output_masks_dir);

} // namespace trtmc::cli
