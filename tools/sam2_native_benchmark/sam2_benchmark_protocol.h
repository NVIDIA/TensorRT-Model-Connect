/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam2/sam2_qualification_authority.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::sam2::benchmark {

// Schema v2 adds the mode-specific regular -> exclusive pre-W3 Q3 receipt
// binding. Consumers must not interpret v1 as providing this lineage.
inline constexpr std::int32_t kBenchmarkReceiptSchemaVersion = 2;
inline constexpr std::size_t kQualificationReplayCount = 3U;
inline constexpr std::size_t kWarmupRowCount = 3U;
inline constexpr std::size_t kMeasurementRowCount = 100U;
// Native runtime admission compares the unrounded metrics from every replay.
// The separately sealed Python source-capture provenance contract is not this
// policy and intentionally retains its reviewed historical thresholds.
inline constexpr double kMinimumFrameMaskIou = kNativeMinimumFrameMaskIou;
inline constexpr double kMinimumMacroMaskIou = kNativeMinimumMacroMaskIou;
inline constexpr double kMinimumGlobalMaskIou = kNativeMinimumGlobalMaskIou;
inline constexpr double kMinimumBboxIou = kNativeMinimumBboxIou;
inline constexpr double kMaximumBboxCoordinateError = kNativeMaximumBboxCoordinateError;
inline constexpr double kMaximumBboxScoreError = kNativeMaximumBboxScoreError;
inline constexpr std::string_view kBaselineReceiptSha256 =
    "af85ed2de2143db6fdf3af40d3621e7f33b97c632904950ee79f3d5f7219f028";
inline constexpr std::string_view kBaselineCaptureScriptSha256 =
    "ba57e65432945f3d6fdafae564a3901f0dd5479ba73e400a5b0a88aa595c30b4";

inline constexpr std::array<std::string_view, 5> kEncodedJpegSha256 = {
    "8a398f40747d5053cfc0d47d45090f2070a10afa4722e7d5b827a6ad0825a5aa",
    "2871555bca47da7473762ca87314b17bd55d100a0f982f78d6449080ff86856f",
    "5594181db7dd1c5da3ce05b945f74e66a5d8d098d71a7cb9e5e43834a393bbe2",
    "c3abc03371458939d09faf331749c2a87cc6fc91128eaab3901b179adb096a35",
    "3d8ea6042c82e7b340277c00666c4c2cefbae5de265ef06a71fe964905ed720b",
};

inline constexpr std::array<std::string_view, 5> kDecodedJpegSha256 = {
    "0bcadde0e5a6f8ba04f79c44f064c5b00d3cd1b250e2f2f3bbf10ef0630a9ce9",
    "0abfd57f9e3886a8c3068bf6bcc353b26d1e3a8a43819a80dfeb00f309b24ec3",
    "9166cc263c3edb262065fa3b98ee062cbf6d781dd656bae13def7f4141b7d025",
    "77525faadfc8a607e4e1556135887caaddd0b64d7cd677fcf47c38ecf9e25a4f",
    "cb0801b490ba13dfb6d36aeef06b049ff67ff11864ef62ccd858a0096d97c6af",
};

struct TimingRow {
    std::size_t index{0U};
    std::uint64_t prefill_nanoseconds{0U};
    std::uint64_t tracker_nanoseconds{0U};
    std::uint64_t total_nanoseconds{0U};
};

struct MetricSummary {
    double mean_milliseconds{0.0};
    double median_milliseconds{0.0};
    double p90_milliseconds{0.0};
    double min_milliseconds{0.0};
    double max_milliseconds{0.0};
};

struct TimingSummary {
    MetricSummary prefill;
    MetricSummary tracker;
    MetricSummary total;
};

struct AccuracyReplay {
    std::size_t index{0U};
    std::string mask_sha256;
    std::string bbox_sha256;
    std::array<std::uint64_t, 5> foreground_pixels{};
    std::array<double, 5> frame_iou{};
    double macro_iou{0.0};
    double global_iou{0.0};
    double bbox_iou{0.0};
    double bbox_max_coordinate_error{0.0};
    double bbox_score_error{0.0};
    bool bbox_label_exact{false};
    std::int32_t candidate_bbox_label{-1};
    double candidate_bbox_score{0.0};
    std::array<double, 4> candidate_bbox_xyxy{};
    std::int32_t reference_bbox_label{-1};
    double reference_bbox_score{0.0};
    std::array<double, 4> reference_bbox_xyxy{};
    bool passes{false};
};

struct AssetFacts {
    std::string checkpoint_sha256;
    std::string source_config_sha256;
    std::string golden_manifest_sha256;
    std::string golden_masks_sha256;
    std::string baseline_receipt_sha256;
    std::string baseline_capture_script_sha256;
    // Set only on the final regular receipt. It binds the exact exclusive Q3
    // receipt published before the first warmup for this same process/bundle.
    std::string q3_receipt_sha256;
    std::uint64_t q3_receipt_size_bytes{0U};
    std::array<std::string, 5> encoded_jpeg_sha256;
    std::array<std::string, 5> decoded_jpeg_sha256;
    std::string native_bundle_sha256;
    std::string native_build_receipt_sha256;
    std::array<std::string, 6> native_plan_sha256;
    std::string benchmark_executable_sha256;
    std::string benchmark_source_manifest_sha256;
    std::string benchmark_source_closure_sha256;
};

struct RuntimeFacts {
    std::int32_t gpu_device{-1};
    std::string gpu_name;
    std::int32_t compute_major{0};
    std::int32_t compute_minor{0};
    std::uint64_t global_memory_bytes{0U};
    std::string tensorrt_version;
    std::string tensorrt_abi;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
    std::string hostname;
    std::string started_at_utc;
    std::string ended_at_utc;
    std::string gpu_uuid;
    std::string pci_bus_id;
    std::string cxx_compiler_id;
    std::string cxx_compiler_version;
    std::int64_t cxx_language_standard{0};
    std::string engine_profiling_verbosity;
    std::string execution_context_nvtx_verbosity;
};

struct ImageAttentionFacts {
    std::string implementation;
    std::string operator_name;
    std::string api;
    std::int32_t block_count{0};
    std::int32_t head_dimension{0};
    std::string query_form;
    std::string key_value_form;
    std::string output_form;
    std::string normalization;
    std::string causal_mask;
    bool decomposable{true};
    bool fused_kernel_intent{false};
    std::string metadata_prefix;
    std::int32_t metadata_index_width{0};
    std::string q_scale_formula;
    std::string k_scale_formula;
    std::string effective_score_scale;
    std::string scale_dtype;
};

enum class BenchmarkMode {
    kDiagnosticBenchmark,
    kAccuracyOnly,
};

struct BenchmarkReceipt {
    BenchmarkMode mode{BenchmarkMode::kDiagnosticBenchmark};
    AssetFacts assets;
    RuntimeFacts runtime;
    ImageAttentionFacts image_attention;
    std::vector<AccuracyReplay> accuracy_only_replays;
    std::vector<AccuracyReplay> prequalification;
    std::vector<TimingRow> warmup_rows;
    std::vector<TimingRow> measurement_rows;
    AccuracyReplay postqualification;
};

TimingSummary summarizeTimingRows(const std::vector<TimingRow>& rows);
std::string makeCanonicalBenchmarkReceipt(const BenchmarkReceipt& receipt);

// Durably publish one completed canonical receipt without following or
// overwriting the requested destination. Publication uses an fsynced
// same-directory temporary inode, authenticated no-replace linking, temporary
// unlink, and parent-directory fsync. Failure rolls back only this call's exact
// published inode and never removes a concurrent replacement.
void writeReceiptExclusive(const std::filesystem::path& path, const std::string& contents);

} // namespace trtmc::sam2::benchmark
