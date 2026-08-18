/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/track_hoi_benchmark.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace trtmc::cli {
namespace {

std::string json_escape(const std::string& text) {
    std::ostringstream output;
    for (unsigned char character : text) {
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(character) << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
            break;
        }
    }
    return output.str();
}

std::filesystem::path normalized_path(const std::string& path) {
    std::error_code error;
    auto normalized = std::filesystem::weakly_canonical(path, error);
    if (!error)
        return normalized;
    error.clear();
    normalized = std::filesystem::absolute(path, error);
    return error ? std::filesystem::path(path).lexically_normal() : normalized.lexically_normal();
}

std::filesystem::path mask_output_path(const std::filesystem::path& root, std::size_t frame_index) {
    std::ostringstream filename;
    filename << "frame_" << std::setw(6) << std::setfill('0') << frame_index << ".npy";
    return normalized_path((root / filename.str()).string());
}

} // namespace

std::size_t track_hoi_p90_index(std::size_t sample_count) {
    if (sample_count == 0U)
        throw std::invalid_argument("track-hoi benchmark needs at least one timed sample");

    // Compute 9 * (N - 1) / 10 without overflowing size_t, then implement
    // Python's round-half-to-even rule from the exact integer remainder.
    const std::size_t intervals = sample_count - 1U;
    std::size_t index = (intervals / 10U) * 9U + ((intervals % 10U) * 9U) / 10U;
    const std::size_t remainder = ((intervals % 10U) * 9U) % 10U;
    if (remainder > 5U || (remainder == 5U && index % 2U != 0U))
        ++index;
    return index;
}

TrackHoiLatencySummary summarize_track_hoi_latency(const std::vector<double>& samples_ms) {
    if (samples_ms.empty())
        throw std::invalid_argument("track-hoi benchmark needs at least one timed sample");

    TrackHoiLatencySummary summary;
    summary.mean_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
                      static_cast<double>(samples_ms.size());
    auto sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2U;
    summary.median_ms =
        sorted.size() % 2U == 0U ? (sorted[middle - 1U] + sorted[middle]) / 2.0 : sorted[middle];
    summary.p90_ms = sorted[track_hoi_p90_index(sorted.size())];
    summary.min_ms = sorted.front();
    summary.max_ms = sorted.back();
    return summary;
}

void validate_track_hoi_output_paths(const std::string& output_json,
                                     const std::string& output_masks_dir,
                                     const std::string& benchmark_json,
                                     std::size_t input_frame_count) {
    const auto accuracy_path = normalized_path(output_json);
    const auto mask_root = normalized_path(output_masks_dir);
    const auto receipt_path =
        benchmark_json.empty() ? std::filesystem::path{} : normalized_path(benchmark_json);
    if (!receipt_path.empty() && receipt_path == accuracy_path) {
        throw std::invalid_argument(
            "track-hoi benchmark receipt must differ from the accuracy output JSON");
    }
    if (accuracy_path == mask_root || (!receipt_path.empty() && receipt_path == mask_root)) {
        throw std::invalid_argument(
            "track-hoi JSON outputs must not replace the mask output directory");
    }
    for (std::size_t frame_index = 0; frame_index < input_frame_count; ++frame_index) {
        const auto mask_path = mask_output_path(mask_root, frame_index);
        if (accuracy_path == mask_path) {
            throw std::invalid_argument(
                "track-hoi accuracy output JSON must not replace a generated mask");
        }
        if (!receipt_path.empty() && receipt_path == mask_path) {
            throw std::invalid_argument(
                "track-hoi benchmark receipt must not replace a generated mask");
        }
    }
}

std::string render_track_hoi_benchmark_receipt(
    const std::string& benchmark_scope, int warmup, std::size_t input_frame_count,
    int32_t produced_frame_count, const std::vector<double>& samples_ms,
    const TrackHoiLatencySummary& summary, const std::string& frame_decode_mode,
    std::size_t frame_decode_max_concurrency, const std::string& output_json,
    const std::string& output_masks_dir) {
    if (frame_decode_mode != "serial" && frame_decode_mode != "model_batch") {
        throw std::invalid_argument("track-hoi benchmark received an invalid frame decode mode");
    }
    if (frame_decode_max_concurrency == 0U ||
        (frame_decode_mode == "serial" && frame_decode_max_concurrency != 1U)) {
        throw std::invalid_argument(
            "track-hoi benchmark received invalid frame decode concurrency");
    }
    const double requests_per_second = summary.mean_ms > 0.0 ? 1000.0 / summary.mean_ms : 0.0;
    const double input_frames_per_second =
        requests_per_second * static_cast<double>(input_frame_count);
    const double produced_frames_per_second =
        requests_per_second * static_cast<double>(produced_frame_count);
    const double five_frame_clips_per_second = produced_frames_per_second / 5.0;
    const bool loaded_request = benchmark_scope == "loaded-request";

    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\n  \"schema_version\": \"trtmc.command-benchmark/v1\",\n"
           << "  \"command\": \"track-hoi\",\n"
           << "  \"benchmark_scope\": \"" << benchmark_scope << "\",\n"
           << "  \"clock\": \"std::chrono::steady_clock_wall\",\n"
           << "  \"warmup_iterations\": " << warmup << ",\n"
           << "  \"timed_iterations\": " << samples_ms.size() << ",\n"
           << "  \"input_frame_count\": " << input_frame_count << ",\n"
           << "  \"produced_frame_count\": " << produced_frame_count << ",\n"
           << "  \"samples\": [\n";
    for (std::size_t index = 0; index < samples_ms.size(); ++index) {
        output << "    {\"iteration\": " << index + 1U << ", \"latency_ms\": " << samples_ms[index]
               << "}" << (index + 1U == samples_ms.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"summary_ms\": {\"mean\": " << summary.mean_ms
           << ", \"median\": " << summary.median_ms << ", \"p90\": " << summary.p90_ms
           << ", \"min\": " << summary.min_ms << ", \"max\": " << summary.max_ms << "},\n"
           << "  \"throughput\": {\"requests_per_second\": " << requests_per_second
           << ", \"five_frame_clips_per_second\": " << five_frame_clips_per_second
           << ", \"input_frames_per_second\": " << input_frames_per_second
           << ", \"produced_frames_per_second\": " << produced_frames_per_second << "},\n"
           << "  \"frame_loading\": {\"frame_decode_mode\": \"" << frame_decode_mode
           << "\", \"frame_decode_max_concurrency\": " << frame_decode_max_concurrency << "},\n"
           << "  \"timing_boundary\": {\n"
           << "    \"pipeline_load_included\": false,\n"
           << "    \"bundle_load_included\": false,\n"
           << "    \"model_load_included\": false,\n"
           << "    \"frame_enumeration_included\": " << (loaded_request ? "true" : "false") << ",\n"
           << "    \"frame_decode_included\": " << (loaded_request ? "true" : "false") << ",\n"
           << "    \"decoded_views_reused\": " << (loaded_request ? "false" : "true") << ",\n"
           << "    \"capability_preprocess_included\": true,\n"
           << "    \"capability_inference_included\": true,\n"
           << "    \"capability_postprocess_included\": true,\n"
           << "    \"request_input_release_included\": false,\n"
           << "    \"output_serialization_included\": false,\n"
           << "    \"warmup_included\": false,\n"
           << "    \"final_materialized_run_included\": false,\n"
           << "    \"benchmark_receipt_write_included\": false,\n"
           << "    \"track_video_return_is_synchronous_boundary\": true\n"
           << "  },\n"
           << "  \"materialized_output\": {\"timed\": false, \"produced_frame_count\": "
           << produced_frame_count << ", \"json_path\": \"" << json_escape(output_json)
           << "\", \"masks_directory\": \"" << json_escape(output_masks_dir) << "\"}\n"
           << "}\n";
    return output.str();
}

} // namespace trtmc::cli
