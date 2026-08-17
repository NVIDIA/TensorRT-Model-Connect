/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "tools/sam2_native_benchmark/sam2_benchmark_accuracy.h"
#include "tools/sam2_native_benchmark/sam2_benchmark_protocol.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

namespace benchmark = trtmc::sam2::benchmark;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename Exception, typename Function>
void checkThrows(Function&& function, const char* needle, const char* message) {
    static_assert(std::is_base_of<std::exception, Exception>::value);
    try {
        function();
    } catch (const Exception& error) {
        if (std::strstr(error.what(), needle) != nullptr)
            return;
        std::cerr << "FAIL: " << message << " (wrong message: " << error.what() << ")\n";
        std::exit(1);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong exception: " << error.what() << ")\n";
        std::exit(1);
    }
    std::cerr << "FAIL: " << message << " (no exception)\n";
    std::exit(1);
}

benchmark::AccuracyReplay accuracy(std::size_t index) {
    benchmark::AccuracyReplay result;
    result.index = index;
    result.mask_sha256 = std::string(64U, 'a');
    result.bbox_sha256 = std::string(64U, 'b');
    result.foreground_pixels = {3600U, 3420U, 4071U, 3770U, 3603U};
    result.frame_iou.fill(0.999);
    result.macro_iou = 0.999;
    result.global_iou = 0.999;
    result.bbox_iou = 0.999;
    result.bbox_max_coordinate_error = 0.125;
    result.bbox_score_error = 0.001;
    result.bbox_label_exact = true;
    result.candidate_bbox_label = 1;
    result.candidate_bbox_score = 0.912345678;
    result.candidate_bbox_xyxy = {100.125, 200.25, 300.5, 400.75};
    result.reference_bbox_label = 1;
    result.reference_bbox_score = 0.912;
    result.reference_bbox_xyxy = {100.0, 200.0, 300.5, 400.75};
    result.passes = true;
    return result;
}

benchmark::BenchmarkReceipt validReceipt() {
    benchmark::BenchmarkReceipt result;
    result.assets.checkpoint_sha256 = std::string(64U, '0');
    result.assets.source_config_sha256 = std::string(64U, '1');
    result.assets.golden_manifest_sha256 = std::string(64U, '2');
    result.assets.golden_masks_sha256 = std::string(64U, '3');
    result.assets.baseline_receipt_sha256 = benchmark::kBaselineReceiptSha256;
    result.assets.baseline_capture_script_sha256 = benchmark::kBaselineCaptureScriptSha256;
    result.assets.q3_receipt_sha256 = std::string(64U, 'f');
    result.assets.q3_receipt_size_bytes = 4096U;
    for (std::size_t index = 0; index < result.assets.encoded_jpeg_sha256.size(); ++index) {
        result.assets.encoded_jpeg_sha256[index] = benchmark::kEncodedJpegSha256[index];
        result.assets.decoded_jpeg_sha256[index] = benchmark::kDecodedJpegSha256[index];
    }
    result.assets.native_bundle_sha256 = std::string(64U, '4');
    result.assets.native_build_receipt_sha256 = std::string(64U, '5');
    constexpr std::array<char, 6> plan_digest_digits = {'6', '7', '8', '9', 'a', 'b'};
    for (std::size_t index = 0; index < result.assets.native_plan_sha256.size(); ++index)
        result.assets.native_plan_sha256[index] = std::string(64U, plan_digest_digits[index]);
    result.assets.benchmark_executable_sha256 = std::string(64U, 'c');
    result.assets.benchmark_source_manifest_sha256 = std::string(64U, 'd');
    result.assets.benchmark_source_closure_sha256 = std::string(64U, 'e');

    result.runtime.gpu_device = 0;
    result.runtime.gpu_name = "NVIDIA L4";
    result.runtime.compute_major = 8;
    result.runtime.compute_minor = 9;
    result.runtime.global_memory_bytes = UINT64_C(24146608128);
    result.runtime.tensorrt_version = "11.1.0.106";
    result.runtime.tensorrt_abi = "11.1";
    result.runtime.cuda_runtime_version = "13.1.0";
    result.runtime.cuda_driver_version = "13.0.0";
    result.runtime.hostname = "ipp2-2249";
    result.runtime.started_at_utc = "2026-08-16T12:00:00Z";
    result.runtime.ended_at_utc = "2026-08-16T12:01:00Z";
    result.runtime.gpu_uuid = "GPU-00000000-0000-0000-0000-000000000000";
    result.runtime.pci_bus_id = "0000:00:00.0";
    result.runtime.cxx_compiler_id = "GNU";
    result.runtime.cxx_compiler_version = "13.3.0";
    result.runtime.cxx_language_standard = 201703L;
    result.runtime.engine_profiling_verbosity = "detailed";
    result.runtime.execution_context_nvtx_verbosity = "none";

    result.image_attention.implementation = "tensorrt_iattention_v2";
    result.image_attention.operator_name = "IAttention";
    result.image_attention.api = "addAttentionV2";
    result.image_attention.block_count = 16;
    result.image_attention.head_dimension = 96;
    result.image_attention.query_form = "padded_bhnd";
    result.image_attention.key_value_form = "padded_bhnd";
    result.image_attention.output_form = "padded_bhnd";
    result.image_attention.normalization = "softmax";
    result.image_attention.causal_mask = "none";
    result.image_attention.decomposable = false;
    result.image_attention.fused_kernel_intent = true;
    result.image_attention.metadata_prefix = "trtmc.sam2.iattention.block.";
    result.image_attention.metadata_index_width = 2;
    result.image_attention.q_scale_formula = "1/sqrt(head_dimension)";
    result.image_attention.k_scale_formula = "none";
    result.image_attention.effective_score_scale = "1/sqrt(head_dimension)";
    result.image_attention.scale_dtype = "bf16";

    for (std::size_t index = 0; index < benchmark::kQualificationReplayCount; ++index)
        result.prequalification.push_back(accuracy(index));
    result.postqualification = accuracy(0U);
    for (std::size_t index = 0; index < benchmark::kWarmupRowCount; ++index) {
        const std::uint64_t prefill = UINT64_C(1000000) + index;
        const std::uint64_t tracker = UINT64_C(2000000) + 2U * index;
        result.warmup_rows.push_back({index, prefill, tracker, prefill + tracker});
    }
    for (std::size_t index = 0; index < benchmark::kMeasurementRowCount; ++index) {
        const std::uint64_t prefill = UINT64_C(1000000) + index;
        const std::uint64_t tracker = UINT64_C(2000000) + 2U * index;
        result.measurement_rows.push_back({index, prefill, tracker, prefill + tracker});
    }
    return result;
}

benchmark::BenchmarkReceipt validAccuracyOnlyReceipt() {
    auto result = validReceipt();
    result.mode = benchmark::BenchmarkMode::kAccuracyOnly;
    result.assets.baseline_receipt_sha256.clear();
    result.assets.baseline_capture_script_sha256.clear();
    result.assets.q3_receipt_sha256.clear();
    result.assets.q3_receipt_size_bytes = 0U;
    result.accuracy_only_replays = std::move(result.prequalification);
    result.warmup_rows.clear();
    result.measurement_rows.clear();
    result.postqualification = {};
    return result;
}

void checkSemanticAccuracyBoundaries() {
    auto boundary = validAccuracyOnlyReceipt();
    for (auto& replay : boundary.accuracy_only_replays) {
        replay.frame_iou.fill(benchmark::kMinimumFrameMaskIou);
        replay.macro_iou = benchmark::kMinimumMacroMaskIou;
        replay.global_iou = benchmark::kMinimumGlobalMaskIou;
        replay.bbox_iou = benchmark::kMinimumBboxIou;
        replay.bbox_max_coordinate_error = benchmark::kMaximumBboxCoordinateError;
        replay.bbox_score_error = benchmark::kMaximumBboxScoreError;
    }
    check(!benchmark::makeCanonicalBenchmarkReceipt(boundary).empty(),
          "semantic accuracy metrics accept their exact boundaries");

    const auto check_below = [](auto mutate, const char* message) {
        auto invalid = validAccuracyOnlyReceipt();
        mutate(invalid.accuracy_only_replays[1]);
        checkThrows<std::runtime_error>(
            [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); },
            "semantic accuracy gate", message);
    };
    check_below(
        [](benchmark::AccuracyReplay& replay) {
            replay.frame_iou[3] = std::nextafter(benchmark::kMinimumFrameMaskIou, 0.0);
        },
        "frame IoU compares the raw value below its boundary");
    check_below(
        [](benchmark::AccuracyReplay& replay) {
            replay.macro_iou = std::nextafter(benchmark::kMinimumMacroMaskIou, 0.0);
        },
        "macro IoU compares the raw value below its boundary");
    check_below(
        [](benchmark::AccuracyReplay& replay) {
            replay.global_iou = std::nextafter(benchmark::kMinimumGlobalMaskIou, 0.0);
        },
        "global IoU compares the raw value below its boundary");
    check_below(
        [](benchmark::AccuracyReplay& replay) {
            replay.bbox_iou = std::nextafter(benchmark::kMinimumBboxIou, 0.0);
        },
        "bbox IoU compares the raw value below its boundary");
    check_below(
        [](benchmark::AccuracyReplay& replay) {
            replay.bbox_max_coordinate_error = std::nextafter(
                benchmark::kMaximumBboxCoordinateError, std::numeric_limits<double>::infinity());
        },
        "bbox coordinate error compares the raw value above its boundary");
    check_below(
        [](benchmark::AccuracyReplay& replay) {
            replay.bbox_score_error = std::nextafter(benchmark::kMaximumBboxScoreError,
                                                     std::numeric_limits<double>::infinity());
        },
        "bbox score error compares the raw value above its boundary");
    check_below([](benchmark::AccuracyReplay& replay) { replay.bbox_label_exact = false; },
                "bbox label remains exact");
}

void checkAccuracyEvaluationBoundaries() {
    constexpr std::size_t frame_pixels =
        static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
        trtmc::sam2::kOriginalImageWidth;
    constexpr std::size_t reference_foreground = 10000U;

    benchmark::GoldenEvidence golden;
    golden.masks.assign(static_cast<std::size_t>(trtmc::sam2::kFrameCount) * frame_pixels, 0U);
    golden.bbox_original_xyxy = {100.0F, 200.0F, 300.0F, 400.0F};
    golden.bbox_score = 0.9F;
    golden.bbox_label = 1;
    for (std::size_t frame = 0; frame < static_cast<std::size_t>(trtmc::sam2::kFrameCount);
         ++frame) {
        const auto begin = golden.masks.begin() + static_cast<std::ptrdiff_t>(frame * frame_pixels);
        std::fill(begin, begin + static_cast<std::ptrdiff_t>(reference_foreground),
                  static_cast<std::uint8_t>(1));
    }

    trtmc::Sam2VideoPromptResult prompt;
    prompt.track.prompt_box_xyxy = golden.bbox_original_xyxy;
    prompt.track.detector_score = golden.bbox_score;
    prompt.track.label = golden.bbox_label;

    const auto candidate = [&](std::size_t extra_frame_zero_deletion) {
        trtmc::Sam2VideoFrameResults results;
        for (std::size_t frame = 0; frame < results.size(); ++frame) {
            results[frame].frame_index = static_cast<std::int32_t>(frame);
            results[frame].height = trtmc::sam2::kOriginalImageHeight;
            results[frame].width = trtmc::sam2::kOriginalImageWidth;
            std::vector<std::uint8_t> mask(frame_pixels, 0U);
            const std::size_t deletions = frame == 0U ? 200U + extra_frame_zero_deletion : 75U;
            std::fill(mask.begin(),
                      mask.begin() + static_cast<std::ptrdiff_t>(reference_foreground - deletions),
                      static_cast<std::uint8_t>(1));
            results[frame].mask = trtmc::Sam2VideoMaskBuffer::host(std::move(mask));
        }
        return results;
    };

    auto boundary_results = candidate(0U);
    const auto boundary = benchmark::evaluateAccuracy(0U, prompt, boundary_results, golden);
    check(boundary.passes && boundary.frame_iou[0] == benchmark::kMinimumFrameMaskIou &&
              boundary.macro_iou == benchmark::kMinimumMacroMaskIou &&
              boundary.global_iou == benchmark::kMinimumGlobalMaskIou,
          "accuracy evaluator accepts exact unrounded mask IoU boundaries");

    auto below_results = candidate(1U);
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::evaluateAccuracy(1U, prompt, below_results, golden); },
        "semantic accuracy gate",
        "accuracy evaluator rejects a one-pixel raw-IoU breach without rounding");
}

void checkAccuracyFailureEvidence() {
    constexpr std::size_t replay_index = 7U;
    constexpr std::size_t frame_pixels =
        static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
        trtmc::sam2::kOriginalImageWidth;

    benchmark::GoldenEvidence golden;
    golden.masks.assign(static_cast<std::size_t>(trtmc::sam2::kFrameCount) * frame_pixels, 0U);
    golden.bbox_original_xyxy = {100.0F, 200.0F, 300.0F, 400.0F};
    golden.bbox_score = 0.9F;
    golden.bbox_label = 1;

    trtmc::Sam2VideoPromptResult prompt;
    prompt.track.prompt_box_xyxy = golden.bbox_original_xyxy;
    prompt.track.detector_score = golden.bbox_score;
    prompt.track.label = 2;

    trtmc::Sam2VideoFrameResults results;
    for (std::size_t frame = 0; frame < results.size(); ++frame) {
        results[frame].frame_index = static_cast<std::int32_t>(frame);
        results[frame].height = trtmc::sam2::kOriginalImageHeight;
        results[frame].width = trtmc::sam2::kOriginalImageWidth;
        results[frame].mask = trtmc::Sam2VideoMaskBuffer::host(
            std::vector<std::uint8_t>(frame_pixels, static_cast<std::uint8_t>(0)));
    }

    try {
        (void)benchmark::evaluateAccuracy(replay_index, prompt, results, golden);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        check(message.find("\"replay_index\":7") != std::string::npos,
              "accuracy failure reports the replay index");
        check(message.find("\"every_frame_iou_min\":0.98") != std::string::npos,
              "accuracy failure reports the frame IoU threshold");
        check(message.find("\"bbox_label_exact\":true") != std::string::npos,
              "accuracy failure reports the exact-label threshold");
        check(message.find("\"frame_iou\"") != std::string::npos,
              "accuracy failure reports the candidate frame IoUs");
        check(message.find("\"bbox_label_exact\":false") != std::string::npos,
              "accuracy failure reports the failing candidate label metric");
        return;
    }
    check(false, "semantic accuracy failure must throw");
}

} // namespace

int main() {
    auto receipt = validReceipt();
    const auto summary = benchmark::summarizeTimingRows(receipt.measurement_rows);
    check(std::abs(summary.prefill.mean_milliseconds - 1.0000495) < 1e-12,
          "prefill mean uses all 100 rows");
    check(std::abs(summary.prefill.median_milliseconds - 1.0000495) < 1e-12,
          "prefill median averages the middle pair");
    check(std::abs(summary.prefill.p90_milliseconds - 1.000089) < 1e-12,
          "p90 uses baseline-compatible zero-based index 89");
    check(std::abs(summary.prefill.min_milliseconds - 1.0) < 1e-12, "minimum is retained");
    check(std::abs(summary.prefill.max_milliseconds - 1.000099) < 1e-12, "maximum is retained");

    const std::string first = benchmark::makeCanonicalBenchmarkReceipt(receipt);
    const std::string second = benchmark::makeCanonicalBenchmarkReceipt(receipt);
    check(first == second && !first.empty() && first.back() == '\n',
          "canonical receipt is deterministic and newline terminated");
    const auto parsed = nlohmann::json::parse(first);
    check(parsed.at("schema_version").get<std::int32_t>() ==
                  benchmark::kBenchmarkReceiptSchemaVersion &&
              parsed.at("mode").get<std::string>() == "diagnostic_benchmark" &&
              !parsed.at("accuracy_only").get<bool>() && parsed.at("timing_performed").get<bool>(),
          "regular receipt explicitly identifies v2 timed diagnostic mode");
    check(parsed.at("timing").at("measurement_rows").size() == 100U, "all N100 rows are retained");
    check(parsed.at("assets").at("q3_receipt_sha256").get<std::string>() ==
                  receipt.assets.q3_receipt_sha256 &&
              parsed.at("assets").at("q3_receipt_size_bytes").get<std::uint64_t>() == 4096U,
          "regular receipt binds the exclusive pre-W3 Q3 receipt");
    check(parsed.at("timing").at("sample_count").get<std::size_t>() == 100U,
          "summary reports n=100");
    check(parsed.at("timing").at("excluded_rows").get<std::size_t>() == 0U,
          "summary reports excluded=0");
    check(parsed.at("accuracy")
                  .at("prequalification")
                  .at(0)
                  .at("candidate_bbox")
                  .at("label")
                  .get<std::int32_t>() == 1,
          "receipt retains candidate bbox values");
    const auto& timing_boundaries = parsed.at("timing_boundaries");
    const auto native_prefill = timing_boundaries.at("native_prefill").get<std::string>();
    const auto native_tracker = timing_boundaries.at("native_tracker").get<std::string>();
    const auto encoded_input = timing_boundaries.at("encoded_input").get<std::string>();
    check(native_prefill.find("decodeSam2JpegBytes") != std::string::npos &&
              native_prefill.find("stable HWC RGB8 frame buffers") != std::string::npos &&
              native_prefill.find("frame 0: same-stream RGB8 H2D") != std::string::npos &&
              native_prefill.find("CUDA Pillow horizontal uint8 pass") != std::string::npos &&
              native_prefill.find("FP32 NCHW normalization") != std::string::npos &&
              native_prefill.find("uint8-to-stable-float") == std::string::npos,
          "receipt binds the RGB8 frame-zero input and CUDA preprocess inside native prefill");
    check(native_tracker.find("frames 1 through 4") != std::string::npos &&
              native_tracker.find("same-stream RGB8 H2D") != std::string::npos &&
              native_tracker.find("CUDA Pillow horizontal uint8 pass") != std::string::npos &&
              native_tracker.find("FP32 NCHW normalization") != std::string::npos,
          "receipt binds the four propagated RGB8 CUDA preprocess stages inside tracker time");
    check(encoded_input.find("byte-for-byte copy into stable HWC RGB8 frame storage") !=
                  std::string::npos &&
              encoded_input.find("no host uint8-to-float conversion") != std::string::npos,
          "receipt describes the decoded RGB8 storage path without a float round trip");
    check(parsed.at("process_model").at("builder_returned_full_bundle_sha256").get<bool>() &&
              parsed.at("process_model")
                  .at("loader_sealed_snapshot_sha256_bound_before_deserialization")
                  .get<bool>() &&
              parsed.at("process_model")
                  .at("receipt_and_plan_evidence_from_builder_not_path_rereads")
                  .get<bool>(),
          "receipt records digest-bound deserialization and builder-returned evidence");
    check(parsed.at("runtime").at("engine_profiling_verbosity").get<std::string>() == "detailed" &&
              parsed.at("runtime").at("execution_context_nvtx_verbosity").get<std::string>() ==
                  "none",
          "receipt records detailed engines with runtime NVTX disabled");
    check(!parsed.at("accuracy").at("repeat_hashes_exact").get<bool>(),
          "receipt treats output hashes as informational");
    check(!parsed.at("accuracy").at("foreground_counts_exact").get<bool>(),
          "receipt treats foreground counts as informational");
    const auto& thresholds = parsed.at("accuracy").at("thresholds");
    check(thresholds.at("every_frame_iou_min").get<double>() == benchmark::kMinimumFrameMaskIou &&
              thresholds.at("macro_iou_min").get<double>() == benchmark::kMinimumMacroMaskIou &&
              thresholds.at("global_iou_min").get<double>() == benchmark::kMinimumGlobalMaskIou &&
              thresholds.at("bbox_iou_min").get<double>() == benchmark::kMinimumBboxIou &&
              thresholds.at("bbox_max_coordinate_error_max").get<double>() ==
                  benchmark::kMaximumBboxCoordinateError &&
              thresholds.at("bbox_score_error_max").get<double>() ==
                  benchmark::kMaximumBboxScoreError &&
              thresholds.at("bbox_label_exact").get<bool>(),
          "receipt exposes the canonical semantic accuracy thresholds");

    auto semantically_repeatable = receipt;
    semantically_repeatable.prequalification[1].mask_sha256 = std::string(64U, 'c');
    semantically_repeatable.prequalification[2].bbox_sha256 = std::string(64U, 'd');
    semantically_repeatable.prequalification[2].foreground_pixels[0] += 1U;
    semantically_repeatable.postqualification.mask_sha256 = std::string(64U, 'e');
    check(!benchmark::makeCanonicalBenchmarkReceipt(semantically_repeatable).empty(),
          "semantic replay gates do not require bitwise-identical hashes or counts");

    auto accuracy_only = validAccuracyOnlyReceipt();
    accuracy_only.accuracy_only_replays[1].mask_sha256 = std::string(64U, 'c');
    accuracy_only.accuracy_only_replays[2].bbox_sha256 = std::string(64U, 'd');
    accuracy_only.accuracy_only_replays[2].foreground_pixels[0] += 1U;
    const auto accuracy_only_json =
        nlohmann::json::parse(benchmark::makeCanonicalBenchmarkReceipt(accuracy_only));
    check(accuracy_only_json.at("mode").get<std::string>() == "accuracy_only" &&
              accuracy_only_json.at("accuracy_only").get<bool>() &&
              !accuracy_only_json.at("timing_performed").get<bool>() &&
              !accuracy_only_json.at("status").at("timing_performed").get<bool>() &&
              !accuracy_only_json.at("status").at("performance_claim").get<bool>() &&
              !accuracy_only_json.at("status").at("runtime_eligible").get<bool>(),
          "accuracy-only receipt makes its non-performance status explicit");
    check(accuracy_only_json.at("accuracy").at("replays").size() ==
                  benchmark::kQualificationReplayCount &&
              accuracy_only_json.at("sequence").at("reset_before_each_replay").get<bool>() &&
              accuracy_only_json.at("sequence").at("frames_per_replay").get<std::size_t>() == 5U,
          "accuracy-only receipt records three reset-separated five-frame replays");
    check(!accuracy_only_json.contains("timing") &&
              !accuracy_only_json.contains("timing_boundaries") &&
              !accuracy_only_json.contains("delivered_baseline_reference") &&
              !accuracy_only_json.at("assets").contains("baseline_receipt_sha256") &&
              !accuracy_only_json.at("assets").contains("baseline_capture_script_sha256"),
          "accuracy-only receipt excludes timing summaries and baseline performance evidence");
    check(accuracy_only_json.at("process_model").at("bundle_build_count").get<int>() == 1 &&
              accuracy_only_json.at("process_model")
                      .at("expected_sha256_bundle_load_count")
                      .get<int>() == 1 &&
              accuracy_only_json.at("assets").at("native_plans").size() == 6U &&
              accuracy_only_json.at("image_attention").at("implementation").get<std::string>() ==
                  "tensorrt_iattention_v2" &&
              accuracy_only_json.at("image_attention").at("operator").get<std::string>() ==
                  "IAttention" &&
              accuracy_only_json.at("image_attention").at("api").get<std::string>() ==
                  "addAttentionV2" &&
              !accuracy_only_json.at("image_attention").at("decomposable").get<bool>() &&
              accuracy_only_json.at("image_attention").at("fused_kernel_intent").get<bool>() &&
              accuracy_only_json.at("image_attention").at("metadata_prefix").get<std::string>() ==
                  "trtmc.sam2.iattention.block." &&
              accuracy_only_json.at("image_attention")
                      .at("metadata_index_width")
                      .get<std::int32_t>() == 2 &&
              accuracy_only_json.at("image_attention").at("q_scale_formula").get<std::string>() ==
                  "1/sqrt(head_dimension)" &&
              accuracy_only_json.at("image_attention").at("k_scale_formula").get<std::string>() ==
                  "none",
          "accuracy-only receipt retains build, expected-hash load, plan, and attention evidence");

    checkAccuracyFailureEvidence();
    checkAccuracyEvaluationBoundaries();
    checkSemanticAccuracyBoundaries();

    auto invalid = receipt;
    ++invalid.measurement_rows.front().total_nanoseconds;
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); }, "row contract",
        "receipt rejects non-adjacent timing boundaries");

    invalid = receipt;
    invalid.image_attention.operator_name = "IMatrixMultiplyLayer";
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); },
        "TensorRT IAttentionV2 contract", "receipt rejects image-attention operator substitution");

    invalid = receipt;
    invalid.image_attention.api = "addMatrixMultiply";
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); },
        "TensorRT IAttentionV2 contract", "receipt rejects image-attention API substitution");

    invalid = receipt;
    invalid.image_attention.scale_dtype = "fp32";
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); },
        "TensorRT IAttentionV2 contract", "receipt rejects image-attention contract substitution");

    invalid = receipt;
    invalid.image_attention.metadata_prefix = "unreviewed.";
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); },
        "TensorRT IAttentionV2 contract", "receipt rejects image-attention metadata drift");

    invalid = receipt;
    invalid.runtime.engine_profiling_verbosity = "layer_names_only";
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); }, "runtime provenance",
        "receipt rejects non-detailed engine profiling verbosity");

    invalid = receipt;
    invalid.runtime.execution_context_nvtx_verbosity = "detailed";
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); }, "runtime provenance",
        "receipt rejects runtime NVTX verbosity overhead");

    invalid = receipt;
    invalid.prequalification[1].passes = false;
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); }, "semantic accuracy gate",
        "receipt rejects a replay that misses the semantic accuracy threshold");

    invalid = validAccuracyOnlyReceipt();
    invalid.warmup_rows.push_back({0U, 1U, 1U, 2U});
    checkThrows<std::runtime_error>(
        [&] { (void)benchmark::makeCanonicalBenchmarkReceipt(invalid); }, "benchmark-only evidence",
        "accuracy-only receipt rejects timing-row evidence");
    return 0;
}
