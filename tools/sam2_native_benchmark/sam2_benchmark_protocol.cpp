/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_benchmark_protocol.h"

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "tools/sam2_native_builder/durable_file_writer.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

namespace trtmc::sam2::benchmark {
namespace {

using Json = nlohmann::ordered_json;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("SAM2 benchmark protocol: " + message);
}

bool isSha256(std::string_view value) {
    return value.size() == 64U && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

void requireSha256(std::string_view value, std::string_view label) {
    if (!isSha256(value))
        fail(std::string(label) + " is not a canonical SHA-256");
}

double milliseconds(std::uint64_t nanoseconds) {
    return static_cast<double>(nanoseconds) / 1000000.0;
}

double receiptNumber(double value, double scale) {
    if (!std::isfinite(value) || value < 0.0)
        fail("receipt contains a negative or non-finite measurement");
    return std::round(value * scale) / scale;
}

template <typename Select>
MetricSummary summarizeMetric(const std::vector<TimingRow>& rows, Select select) {
    std::vector<std::uint64_t> ordered;
    ordered.reserve(rows.size());
    long double sum = 0.0L;
    for (const auto& row : rows) {
        const auto value = select(row);
        ordered.push_back(value);
        sum += static_cast<long double>(value);
    }
    std::sort(ordered.begin(), ordered.end());
    const std::size_t middle = ordered.size() / 2U;
    const long double median_ns =
        ordered.size() % 2U == 0U
            ? (static_cast<long double>(ordered[middle - 1U]) + ordered[middle]) / 2.0L
            : static_cast<long double>(ordered[middle]);
    // The delivered Python baseline uses round((N - 1) * .90). N is fixed at
    // 100, so the comparable sample is the zero-based element 89.
    const std::size_t p90_index = 89U;

    MetricSummary result;
    result.mean_milliseconds = static_cast<double>(sum / rows.size() / 1000000.0L);
    result.median_milliseconds = static_cast<double>(median_ns / 1000000.0L);
    result.p90_milliseconds = milliseconds(ordered[p90_index]);
    result.min_milliseconds = milliseconds(ordered.front());
    result.max_milliseconds = milliseconds(ordered.back());
    return result;
}

Json metricJson(const MetricSummary& summary) {
    Json result;
    result["mean_ms"] = receiptNumber(summary.mean_milliseconds, 1000000.0);
    result["median_ms"] = receiptNumber(summary.median_milliseconds, 1000000.0);
    result["p90_ms"] = receiptNumber(summary.p90_milliseconds, 1000000.0);
    result["min_ms"] = receiptNumber(summary.min_milliseconds, 1000000.0);
    result["max_ms"] = receiptNumber(summary.max_milliseconds, 1000000.0);
    return result;
}

Json timingRowJson(const TimingRow& row) {
    Json result;
    result["index"] = row.index;
    result["native_prefill_ns"] = row.prefill_nanoseconds;
    result["native_tracker_ns"] = row.tracker_nanoseconds;
    result["closest_envelope_total_ns"] = row.total_nanoseconds;
    result["native_prefill_ms"] = receiptNumber(milliseconds(row.prefill_nanoseconds), 1000000.0);
    result["native_tracker_ms"] = receiptNumber(milliseconds(row.tracker_nanoseconds), 1000000.0);
    result["closest_envelope_total_ms"] =
        receiptNumber(milliseconds(row.total_nanoseconds), 1000000.0);
    return result;
}

Json accuracyJson(const AccuracyReplay& replay) {
    requireSha256(replay.mask_sha256, "candidate mask hash");
    requireSha256(replay.bbox_sha256, "candidate bbox hash");
    Json result;
    result["index"] = replay.index;
    result["mask_sha256"] = replay.mask_sha256;
    result["bbox_sha256"] = replay.bbox_sha256;
    result["foreground_pixels"] = replay.foreground_pixels;
    Json frame_iou = Json::array();
    for (const double value : replay.frame_iou)
        frame_iou.push_back(receiptNumber(value, 1000000000.0));
    result["frame_iou"] = std::move(frame_iou);
    result["macro_iou"] = receiptNumber(replay.macro_iou, 1000000000.0);
    result["global_iou"] = receiptNumber(replay.global_iou, 1000000000.0);
    result["bbox_iou"] = receiptNumber(replay.bbox_iou, 1000000000.0);
    result["bbox_max_coordinate_error"] =
        receiptNumber(replay.bbox_max_coordinate_error, 1000000000.0);
    result["bbox_score_error"] = receiptNumber(replay.bbox_score_error, 1000000000.0);
    result["bbox_label_exact"] = replay.bbox_label_exact;
    Json candidate_xyxy = Json::array();
    Json reference_xyxy = Json::array();
    for (std::size_t index = 0; index < replay.candidate_bbox_xyxy.size(); ++index) {
        candidate_xyxy.push_back(receiptNumber(replay.candidate_bbox_xyxy[index], 1000000.0));
        reference_xyxy.push_back(receiptNumber(replay.reference_bbox_xyxy[index], 1000000.0));
    }
    result["candidate_bbox"] = {
        {"label", replay.candidate_bbox_label},
        {"score", receiptNumber(replay.candidate_bbox_score, 1000000000.0)},
        {"original_image_xyxy", std::move(candidate_xyxy)},
    };
    result["reference_bbox"] = {
        {"label", replay.reference_bbox_label},
        {"score", receiptNumber(replay.reference_bbox_score, 1000000000.0)},
        {"original_image_xyxy", std::move(reference_xyxy)},
    };
    result["passes"] = replay.passes;
    return result;
}

void validateRowSet(const std::vector<TimingRow>& rows, std::size_t expected,
                    std::string_view label) {
    if (rows.size() != expected)
        fail(std::string(label) + " row count drifted");
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        if (row.index != index || row.prefill_nanoseconds == 0U || row.tracker_nanoseconds == 0U ||
            row.total_nanoseconds == 0U ||
            row.prefill_nanoseconds >
                std::numeric_limits<std::uint64_t>::max() - row.tracker_nanoseconds ||
            row.total_nanoseconds != row.prefill_nanoseconds + row.tracker_nanoseconds) {
            fail(std::string(label) + " row contract drifted");
        }
    }
}

void validateSemanticReplay(const AccuracyReplay& replay, std::size_t expected_index,
                            std::string_view label) {
    const auto unit_interval = [](double value) {
        return std::isfinite(value) && value >= 0.0 && value <= 1.0;
    };
    if (replay.index != expected_index || !replay.passes ||
        !std::all_of(
            replay.frame_iou.begin(), replay.frame_iou.end(),
            [&](double value) { return unit_interval(value) && value >= kMinimumFrameMaskIou; }) ||
        !unit_interval(replay.macro_iou) || replay.macro_iou < kMinimumMacroMaskIou ||
        !unit_interval(replay.global_iou) || replay.global_iou < kMinimumGlobalMaskIou ||
        !unit_interval(replay.bbox_iou) || replay.bbox_iou < kMinimumBboxIou ||
        !std::isfinite(replay.bbox_max_coordinate_error) ||
        replay.bbox_max_coordinate_error < 0.0 ||
        replay.bbox_max_coordinate_error > kMaximumBboxCoordinateError ||
        !std::isfinite(replay.bbox_score_error) || replay.bbox_score_error < 0.0 ||
        replay.bbox_score_error > kMaximumBboxScoreError || !replay.bbox_label_exact) {
        fail(std::string(label) + " semantic accuracy gate failed");
    }
}

void validateReplaySet(const std::vector<AccuracyReplay>& replays, std::string_view label) {
    if (replays.size() != kQualificationReplayCount)
        fail(std::string(label) + " replay count drifted");
    for (std::size_t index = 0; index < replays.size(); ++index)
        validateSemanticReplay(replays[index], index, label);
}

void validateQualification(const BenchmarkReceipt& receipt) {
    if (receipt.mode == BenchmarkMode::kAccuracyOnly) {
        if (!receipt.prequalification.empty() || !receipt.warmup_rows.empty() ||
            !receipt.measurement_rows.empty() || !receipt.postqualification.mask_sha256.empty() ||
            !receipt.postqualification.bbox_sha256.empty() || receipt.postqualification.passes) {
            fail("accuracy-only receipt contains benchmark-only evidence");
        }
        validateReplaySet(receipt.accuracy_only_replays, "accuracy-only");
        return;
    }
    if (!receipt.accuracy_only_replays.empty())
        fail("diagnostic benchmark receipt contains accuracy-only replay evidence");
    validateReplaySet(receipt.prequalification, "prequalification");
    validateSemanticReplay(receipt.postqualification, 0U, "postqualification");
}

void validateHashes(const BenchmarkReceipt& receipt) {
    const auto& assets = receipt.assets;
    requireSha256(assets.checkpoint_sha256, "checkpoint hash");
    requireSha256(assets.source_config_sha256, "source config hash");
    requireSha256(assets.golden_manifest_sha256, "golden manifest hash");
    requireSha256(assets.golden_masks_sha256, "golden masks hash");
    requireSha256(assets.native_bundle_sha256, "native bundle hash");
    requireSha256(assets.native_build_receipt_sha256, "native build receipt hash");
    requireSha256(assets.benchmark_executable_sha256, "benchmark executable hash");
    requireSha256(assets.benchmark_source_manifest_sha256, "benchmark source manifest hash");
    requireSha256(assets.benchmark_source_closure_sha256, "benchmark source closure hash");
    if (receipt.mode == BenchmarkMode::kAccuracyOnly) {
        if (!assets.baseline_receipt_sha256.empty() ||
            !assets.baseline_capture_script_sha256.empty() || !assets.q3_receipt_sha256.empty() ||
            assets.q3_receipt_size_bytes != 0U) {
            fail("accuracy-only receipt contains baseline performance evidence");
        }
    } else {
        requireSha256(assets.baseline_receipt_sha256, "baseline receipt hash");
        requireSha256(assets.baseline_capture_script_sha256, "baseline capture script hash");
        if (assets.baseline_receipt_sha256 != kBaselineReceiptSha256)
            fail("baseline receipt hash does not match the delivered W3/N100 receipt");
        if (assets.baseline_capture_script_sha256 != kBaselineCaptureScriptSha256)
            fail("baseline capture script hash does not match the reviewed capture tool");
        requireSha256(assets.q3_receipt_sha256, "Q3 receipt hash");
        if (assets.q3_receipt_size_bytes == 0U)
            fail("Q3 receipt size is zero");
    }
    for (const auto& plan_sha256 : assets.native_plan_sha256)
        requireSha256(plan_sha256, "native plan hash");
    for (std::size_t index = 0; index < assets.encoded_jpeg_sha256.size(); ++index) {
        requireSha256(assets.encoded_jpeg_sha256[index], "encoded JPEG hash");
        requireSha256(assets.decoded_jpeg_sha256[index], "decoded JPEG hash");
        if (assets.encoded_jpeg_sha256[index] != kEncodedJpegSha256[index] ||
            assets.decoded_jpeg_sha256[index] != kDecodedJpegSha256[index]) {
            fail("JPEG provenance hash drifted");
        }
    }
}

void validateImageAttention(const ImageAttentionFacts& attention) {
    if (attention.implementation != "tensorrt_iattention_v2" ||
        attention.operator_name != "IAttention" || attention.api != "addAttentionV2" ||
        attention.block_count != 16 || attention.head_dimension != 96 ||
        attention.query_form != "padded_bhnd" || attention.key_value_form != "padded_bhnd" ||
        attention.output_form != "padded_bhnd" || attention.normalization != "softmax" ||
        attention.causal_mask != "none" || attention.decomposable ||
        !attention.fused_kernel_intent ||
        attention.metadata_prefix != trtmc::sam2::kImageAttentionMetadataPrefix ||
        attention.metadata_index_width != trtmc::sam2::kImageAttentionMetadataIndexWidth ||
        attention.q_scale_formula != "1/sqrt(head_dimension)" ||
        attention.k_scale_formula != "none" ||
        attention.effective_score_scale != "1/sqrt(head_dimension)" ||
        attention.scale_dtype != "bf16") {
        fail("image attention does not match the exact TensorRT IAttentionV2 contract");
    }
}

std::string systemError(std::string_view operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

void writeAll(int descriptor, const std::string& contents) {
    std::size_t written = 0U;
    while (written != contents.size()) {
        const auto result =
            ::write(descriptor, contents.data() + written, contents.size() - written);
        if (result < 0) {
            if (errno == EINTR)
                continue;
            fail(systemError("receipt write failed"));
        }
        if (result == 0)
            fail("receipt write made no progress");
        written += static_cast<std::size_t>(result);
    }
}

} // namespace

TimingSummary summarizeTimingRows(const std::vector<TimingRow>& rows) {
    validateRowSet(rows, kMeasurementRowCount, "measurement");
    TimingSummary result;
    result.prefill = summarizeMetric(rows, [](const auto& row) { return row.prefill_nanoseconds; });
    result.tracker = summarizeMetric(rows, [](const auto& row) { return row.tracker_nanoseconds; });
    result.total = summarizeMetric(rows, [](const auto& row) { return row.total_nanoseconds; });
    return result;
}

std::string makeCanonicalBenchmarkReceipt(const BenchmarkReceipt& receipt) {
    const bool accuracy_only = receipt.mode == BenchmarkMode::kAccuracyOnly;
    validateHashes(receipt);
    validateQualification(receipt);
    TimingSummary summary;
    if (!accuracy_only) {
        validateRowSet(receipt.warmup_rows, kWarmupRowCount, "warmup");
        validateRowSet(receipt.measurement_rows, kMeasurementRowCount, "measurement");
        summary = summarizeTimingRows(receipt.measurement_rows);
    }

    const auto& runtime = receipt.runtime;
    if (runtime.gpu_device < 0 || runtime.gpu_name.empty() || runtime.compute_major <= 0 ||
        runtime.compute_minor < 0 || runtime.global_memory_bytes == 0U ||
        runtime.tensorrt_version.empty() || runtime.tensorrt_abi.empty() ||
        runtime.cuda_runtime_version.empty() || runtime.cuda_driver_version.empty() ||
        runtime.hostname.empty() || runtime.started_at_utc.empty() ||
        runtime.ended_at_utc.empty() || runtime.gpu_uuid.empty() || runtime.pci_bus_id.empty() ||
        runtime.cxx_compiler_id.empty() || runtime.cxx_compiler_version.empty() ||
        runtime.cxx_language_standard != 201703L ||
        runtime.engine_profiling_verbosity != trtmc::sam2::kPlanProfilingVerbosity ||
        runtime.execution_context_nvtx_verbosity !=
            trtmc::sam2::kBenchmarkExecutionContextNvtxVerbosity) {
        fail("runtime provenance is incomplete");
    }
    validateImageAttention(receipt.image_attention);

    Json root;
    root["schema_version"] = kBenchmarkReceiptSchemaVersion;
    root["family"] = "sam2";
    root["workload"] = "sam2.1-hiera-small-bbox-five-frame";
    root["mode"] = accuracy_only ? "accuracy_only" : "diagnostic_benchmark";
    root["accuracy_only"] = accuracy_only;
    root["timing_performed"] = !accuracy_only;
    root["status"] = {
        {"accuracy_qualified_for_this_diagnostic", true},
        {"runtime_eligible", false},
        {"performance_claim", false},
        {"timing_performed", !accuracy_only},
        {"outlier_filtering", false},
    };
    Json process_model = {
        {"tensorrt_iattention_v2_image_attention", true},
        {"external_attention_dso_loaded", false},
        {"bundle_build_count", 1},
        {"expected_sha256_bundle_load_count", 1},
        {"builder_returned_full_bundle_sha256", true},
        {"loader_sealed_snapshot_sha256_bound_before_deserialization", true},
        {"receipt_and_plan_evidence_from_builder_not_path_rereads", true},
        {"engine_deserialization_count", 6},
        {"shared_nonblocking_cuda_stream", true},
        {"processor", "makeNativeDeviceVideoProcessor"},
        {"checked_enqueue_v3_adapter", true},
    };
    if (accuracy_only) {
        process_model["checkpoint_graph_build_before_replays"] = true;
        process_model["six_plan_deserializations_before_replays"] = true;
    } else {
        process_model["checkpoint_graph_build_outside_timing"] = true;
        process_model["six_plan_deserializations_outside_timing"] = true;
    }
    root["process_model"] = std::move(process_model);
    if (accuracy_only) {
        root["sequence"] = {
            {"accuracy_replays", kQualificationReplayCount},
            {"frames_per_replay", 5},
            {"reset_before_each_replay", true},
            {"order", "Q3_only"},
            {"warmup_rows", 0},
            {"measurement_rows", 0},
            {"postqualification_replays", 0},
        };
    } else {
        root["sequence"] = {
            {"prequalification_replays", kQualificationReplayCount},
            {"warmup_rows", kWarmupRowCount},
            {"measurement_rows", kMeasurementRowCount},
            {"postqualification_replays", 1},
            {"order", "Q3_then_W3_then_N100_then_Q1"},
            {"accuracy_materialization_between_timing_rows", false},
        };
        root["timing_boundaries"] = {
            {"clock", "std::chrono::steady_clock synchronized wall time"},
            {"reset",
             "t0 -> processor.reset (clear run state, drain workspace, invalidate the completed "
             "run, invoke the reset_execution_context hook on six stable modules without "
             "context recreation, validate device graph, transition to idle) -> "
             "cudaStreamSynchronize; included in native prefill"},
            {"native_prefill",
             "t0 -> processor.reset -> cudaStreamSynchronize -> decodeSam2JpegBytes for all "
             "five retained authenticated byte vectors -> copy decoded RGB8 bytes into five "
             "stable HWC RGB8 frame buffers -> run_bbox_prompt (frame 0: same-stream RGB8 "
             "H2D -> CUDA Pillow horizontal uint8 pass -> CUDA Pillow vertical uint8 pass plus "
             "FP32 NCHW normalization -> image and prompt enqueue) -> cudaStreamSynchronize -> "
             "t1"},
            {"native_tracker",
             "t1 -> propagate frames 1 through 4 (each frame: same-stream RGB8 H2D -> CUDA "
             "Pillow horizontal uint8 pass -> CUDA Pillow vertical uint8 pass plus FP32 NCHW "
             "normalization -> image and recurrent enqueue) -> cudaStreamSynchronize -> t2"},
            {"closest_envelope_total", "t0 -> t2"},
            {"jpeg_file_open_and_read_inside_timing", false},
            {"encoded_input",
             "pre-read authenticated immutable byte vectors; decodeSam2JpegBytes lvalue copy and "
             "JPEG decode plus byte-for-byte copy into stable HWC RGB8 frame storage are inside "
             "each native prefill; no host uint8-to-float conversion is performed"},
            {"accuracy_and_mask_download_inside_timing", false},
            {"native_stage_split_comparable_to_delivered_baseline", false},
            {"total_is_exact_apples_to_apples_with_delivered_lazy_loader", false},
            {"comparison_scope", "closest five-frame end-to-end inference envelope only"},
        };
    }

    Json assets;
    assets["checkpoint_sha256"] = receipt.assets.checkpoint_sha256;
    assets["source_config_sha256"] = receipt.assets.source_config_sha256;
    assets["golden_manifest_sha256"] = receipt.assets.golden_manifest_sha256;
    assets["golden_masks_sha256"] = receipt.assets.golden_masks_sha256;
    if (!accuracy_only) {
        assets["baseline_receipt_sha256"] = receipt.assets.baseline_receipt_sha256;
        assets["baseline_capture_script_sha256"] = receipt.assets.baseline_capture_script_sha256;
        assets["q3_receipt_sha256"] = receipt.assets.q3_receipt_sha256;
        assets["q3_receipt_size_bytes"] = receipt.assets.q3_receipt_size_bytes;
        assets["q3_receipt_role"] =
            "exclusive same-process same-bundle Q3 receipt published before W3";
    }
    assets["encoded_jpeg_sha256"] = receipt.assets.encoded_jpeg_sha256;
    assets["decoded_rgb_sha256"] = receipt.assets.decoded_jpeg_sha256;
    assets["native_bundle_sha256"] = receipt.assets.native_bundle_sha256;
    assets["native_build_receipt_sha256"] = receipt.assets.native_build_receipt_sha256;
    Json native_plans = Json::array();
    for (std::size_t index = 0; index < receipt.assets.native_plan_sha256.size(); ++index) {
        native_plans.push_back({
            {"section", trtmc::sam2::kRequiredPlanSections[index]},
            {"sha256", receipt.assets.native_plan_sha256[index]},
        });
    }
    assets["native_plans"] = std::move(native_plans);
    assets["benchmark_executable_sha256"] = receipt.assets.benchmark_executable_sha256;
    assets["benchmark_source_manifest_sha256"] = receipt.assets.benchmark_source_manifest_sha256;
    assets["benchmark_source_closure_sha256"] = receipt.assets.benchmark_source_closure_sha256;
    assets["benchmark_source_closure_role"] =
        "run-time snapshot of declared repository source and build-control inputs; executable "
        "SHA-256 is authoritative for the binary actually run";
    root["assets"] = std::move(assets);

    root["runtime"] = {
        {"gpu_device", runtime.gpu_device},
        {"gpu_name", runtime.gpu_name},
        {"compute_capability",
         std::to_string(runtime.compute_major) + "." + std::to_string(runtime.compute_minor)},
        {"global_memory_bytes", runtime.global_memory_bytes},
        {"tensorrt_version", runtime.tensorrt_version},
        {"tensorrt_abi", runtime.tensorrt_abi},
        {"cuda_runtime_version", runtime.cuda_runtime_version},
        {"cuda_driver_version", runtime.cuda_driver_version},
        {"hostname", runtime.hostname},
        {"started_at_utc", runtime.started_at_utc},
        {"ended_at_utc", runtime.ended_at_utc},
        {"gpu_uuid", runtime.gpu_uuid},
        {"pci_bus_id", runtime.pci_bus_id},
        {"cxx_compiler_id", runtime.cxx_compiler_id},
        {"cxx_compiler_version", runtime.cxx_compiler_version},
        {"cxx_language_standard", runtime.cxx_language_standard},
        {"engine_profiling_verbosity", runtime.engine_profiling_verbosity},
        {"execution_context_nvtx_verbosity", runtime.execution_context_nvtx_verbosity},
    };

    const auto& attention = receipt.image_attention;
    root["image_attention"] = {
        {"implementation", attention.implementation},
        {"operator", attention.operator_name},
        {"api", attention.api},
        {"block_count", attention.block_count},
        {"head_dimension", attention.head_dimension},
        {"query_form", attention.query_form},
        {"key_value_form", attention.key_value_form},
        {"output_form", attention.output_form},
        {"normalization", attention.normalization},
        {"causal_mask", attention.causal_mask},
        {"decomposable", attention.decomposable},
        {"fused_kernel_intent", attention.fused_kernel_intent},
        {"metadata_prefix", attention.metadata_prefix},
        {"metadata_index_width", attention.metadata_index_width},
        {"q_scale_formula", attention.q_scale_formula},
        {"k_scale_formula", attention.k_scale_formula},
        {"effective_score_scale", attention.effective_score_scale},
        {"scale_dtype", attention.scale_dtype},
    };

    Json accuracy_replays = Json::array();
    const auto& replay_source =
        accuracy_only ? receipt.accuracy_only_replays : receipt.prequalification;
    for (const auto& replay : replay_source)
        accuracy_replays.push_back(accuracyJson(replay));
    Json accuracy = {
        {"thresholds",
         {{"every_frame_iou_min", kMinimumFrameMaskIou},
          {"macro_iou_min", kMinimumMacroMaskIou},
          {"global_iou_min", kMinimumGlobalMaskIou},
          {"bbox_iou_min", kMinimumBboxIou},
          {"bbox_max_coordinate_error_max", kMaximumBboxCoordinateError},
          {"bbox_score_error_max", kMaximumBboxScoreError},
          {"bbox_label_exact", true}}},
        {"repeat_hashes_exact", false},
        {"foreground_counts_exact", false},
        {"repeat_contract",
         "each reset-separated replay independently passes the semantic mask and bbox gates; "
         "hashes and foreground counts are informational"},
    };
    if (accuracy_only) {
        accuracy["replays"] = std::move(accuracy_replays);
    } else {
        Json postqualification = Json::array();
        postqualification.push_back(accuracyJson(receipt.postqualification));
        accuracy["prequalification"] = std::move(accuracy_replays);
        accuracy["postqualification"] = std::move(postqualification);
    }
    root["accuracy"] = std::move(accuracy);

    if (!accuracy_only) {
        Json warmups = Json::array();
        for (const auto& row : receipt.warmup_rows)
            warmups.push_back(timingRowJson(row));
        Json measurements = Json::array();
        for (const auto& row : receipt.measurement_rows)
            measurements.push_back(timingRowJson(row));
        root["timing"] = {
            {"sample_count", kMeasurementRowCount},
            {"excluded_rows", 0},
            {"p90_method", "sorted sample index round((n-1)*0.90), zero-based index 89 for n=100"},
            {"outlier_removal", false},
            {"warmup_rows", std::move(warmups)},
            {"measurement_rows", std::move(measurements)},
            {"summary",
             {{"native_prefill", metricJson(summary.prefill)},
              {"native_tracker", metricJson(summary.tracker)},
              {"closest_envelope_total", metricJson(summary.total)}}},
        };
        root["delivered_baseline_reference"] = {
            {"receipt_sha256", std::string(kBaselineReceiptSha256)},
            {"baseline_receipt_contains_asset_hashes", false},
            {"baseline_asset_binding", "external_reviewed_capture_evidence"},
            {"warmup_rows", 3},
            {"measurement_rows", 100},
            {"total_mean_ms", 257.1344714984298},
            {"total_median_ms", 253.67085821926594},
            {"total_p90_ms", 265.56191593408585},
            {"comparison_warning",
             "raw reference only; no speedup is computed because the delivered baseline uses a "
             "different lazy directory-loader and stage split, explicitly enables matmul and "
             "cuDNN TF32 while the native build disables TF32, and places release_state plus GC "
             "plus empty_cache outside timing while native reset and run invalidation are inside "
             "the closest envelope"},
        };
    }

    return root.dump() + "\n";
}

void writeReceiptExclusive(const std::filesystem::path& path, const std::string& contents) {
    if (path.empty() || !path.has_filename() || contents.empty())
        fail("receipt destination and contents must be nonempty");
    try {
        (void)durable_file::writeExclusiveDurably(
            path, "SAM2 benchmark receipt",
            [&](int descriptor) { writeAll(descriptor, contents); });
    } catch (const durable_file::WriteError& error) {
        fail(error.what());
    }
}

} // namespace trtmc::sam2::benchmark
