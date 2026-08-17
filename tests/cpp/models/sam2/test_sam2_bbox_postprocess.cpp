/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_bbox_postprocess.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void check_equal(float actual, float expected, const char* message) {
    if (actual != expected) {
        std::cerr << "FAIL: " << message << " (actual " << actual << ", expected " << expected
                  << ")\n";
        std::exit(1);
    }
}

template <typename Exception, typename Function>
void check_throws(Function&& function, const char* needle, const char* message) {
    static_assert(std::is_base_of<std::exception, Exception>::value,
                  "test exception must derive from std::exception");
    try {
        function();
    } catch (const Exception& error) {
        if (std::strstr(error.what(), needle) != nullptr)
            return;
        std::cerr << "FAIL: " << message << " (wrong message '" << error.what() << "')\n";
        std::exit(1);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong exception '" << error.what() << "')\n";
        std::exit(1);
    }
    std::cerr << "FAIL: " << message << " (no exception)\n";
    std::exit(1);
}

uint16_t float_to_bfloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest, ties to even. This helper only creates test inputs;
    // production postprocessing consumes the engine's native BF16 bits.
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float reference_sigmoid(float value) {
    if (value >= 0.0F)
        return 1.0F / (1.0F + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

float raw_logit_with_exact_sigmoid(float target) {
    float center = static_cast<float>(
        std::log(static_cast<double>(target) / static_cast<double>(1.0F - target)));
    if (reference_sigmoid(center) == target)
        return center;

    float lower = center;
    float upper = center;
    for (int iteration = 0; iteration < 4096; ++iteration) {
        lower = std::nextafter(lower, -std::numeric_limits<float>::infinity());
        if (reference_sigmoid(lower) == target)
            return lower;
        upper = std::nextafter(upper, std::numeric_limits<float>::infinity());
        if (reference_sigmoid(upper) == target)
            return upper;
    }
    check(false, "synthetic logit can represent the requested exact sigmoid score");
    return 0.0F;
}

struct OwnedRawOutputs {
    static constexpr std::array<int32_t, 3> kStrides{8, 16, 32};
    static constexpr std::array<int32_t, 3> kSizes{128, 64, 32};

    explicit OwnedRawOutputs(trtmc::Sam2BBoxDataType type = trtmc::Sam2BBoxDataType::kFloat32)
        : data_type(type) {
        const std::size_t bytes =
            data_type == trtmc::Sam2BBoxDataType::kFloat32 ? sizeof(float) : sizeof(uint16_t);
        for (std::size_t level = 0; level < kSizes.size(); ++level) {
            const auto area =
                static_cast<std::size_t>(kSizes[level]) * static_cast<std::size_t>(kSizes[level]);
            classification[level].resize(2U * area * bytes);
            regression[level].resize(4U * area * bytes);
            for (std::size_t index = 0; index < 2U * area; ++index)
                write(classification[level], index, -20.0F);
            for (std::size_t index = 0; index < 4U * area; ++index)
                write(regression[level], index, 0.0F);
        }
    }

    void set_candidate(int32_t stride, int32_t y, int32_t x, int32_t label, float logit,
                       const std::array<float, 4>& ltrb) {
        const auto level = level_for_stride(stride);
        const auto size = static_cast<std::size_t>(kSizes[level]);
        const auto area = size * size;
        const auto local_index = static_cast<std::size_t>(y) * size + static_cast<std::size_t>(x);
        write(classification[level], static_cast<std::size_t>(label) * area + local_index, logit);
        for (std::size_t coordinate = 0; coordinate < ltrb.size(); ++coordinate)
            write(regression[level], coordinate * area + local_index, ltrb[coordinate]);
    }

    void set_class_value(int32_t stride, int32_t y, int32_t x, int32_t label, float value) {
        const auto level = level_for_stride(stride);
        const auto size = static_cast<std::size_t>(kSizes[level]);
        const auto area = size * size;
        const auto local_index = static_cast<std::size_t>(y) * size + static_cast<std::size_t>(x);
        write(classification[level], static_cast<std::size_t>(label) * area + local_index, value);
    }

    void set_regression_value(int32_t stride, int32_t channel, int32_t y, int32_t x, float value) {
        const auto level = level_for_stride(stride);
        const auto size = static_cast<std::size_t>(kSizes[level]);
        const auto area = size * size;
        const auto local_index = static_cast<std::size_t>(y) * size + static_cast<std::size_t>(x);
        write(regression[level], static_cast<std::size_t>(channel) * area + local_index, value);
    }

    trtmc::Sam2BBoxRawOutputs views() const {
        trtmc::Sam2BBoxRawOutputs result;
        result.bbox_cls_stride_8 = view(classification[0], 2, kSizes[0]);
        result.bbox_cls_stride_16 = view(classification[1], 2, kSizes[1]);
        result.bbox_cls_stride_32 = view(classification[2], 2, kSizes[2]);
        result.bbox_reg_stride_8 = view(regression[0], 4, kSizes[0]);
        result.bbox_reg_stride_16 = view(regression[1], 4, kSizes[1]);
        result.bbox_reg_stride_32 = view(regression[2], 4, kSizes[2]);
        return result;
    }

    trtmc::Sam2BBoxDataType data_type;
    std::array<std::vector<unsigned char>, 3> classification;
    std::array<std::vector<unsigned char>, 3> regression;

  private:
    std::size_t level_for_stride(int32_t stride) const {
        for (std::size_t index = 0; index < kStrides.size(); ++index) {
            if (kStrides[index] == stride)
                return index;
        }
        check(false, "test fixture uses a supported SAM2 bbox stride");
        return 0;
    }

    void write(std::vector<unsigned char>& storage, std::size_t index, float value) {
        if (data_type == trtmc::Sam2BBoxDataType::kFloat32) {
            std::memcpy(storage.data() + index * sizeof(value), &value, sizeof(value));
            return;
        }
        const uint16_t encoded = float_to_bfloat16(value);
        std::memcpy(storage.data() + index * sizeof(encoded), &encoded, sizeof(encoded));
    }

    trtmc::Sam2BBoxTensorView view(const std::vector<unsigned char>& storage, int64_t channels,
                                   int64_t size) const {
        return {storage.data(),
                data_type,
                {1, channels, size, size},
                static_cast<std::size_t>(channels * size * size)};
    }
};

constexpr std::array<int32_t, 3> OwnedRawOutputs::kStrides;
constexpr std::array<int32_t, 3> OwnedRawOutputs::kSizes;

void test_contract_and_captured_reference_box() {
    check(trtmc::kSam2BBoxModelHeight == 1024 && trtmc::kSam2BBoxModelWidth == 1024,
          "sam2 bbox model canvas is fixed at 1024 square");
    check(trtmc::kSam2BBoxPointOffset == 0.5F && trtmc::kSam2BBoxScoreThreshold == 0.35F &&
              trtmc::kSam2BBoxPreNmsTopK == 100 && trtmc::kSam2BBoxNmsIouThreshold == 0.2F,
          "sam2 bbox source-derived selection constants stay exact");
    check(trtmc::kSam2BBoxConfiguredMaxPerImage == 10 &&
              !trtmc::kSam2BBoxAppliesConfiguredMaxPerImage,
          "sam2 bbox records but deliberately does not apply configured max ten");
    check(!trtmc::kSam2BBoxTieOrderParityQualified,
          "sam2 bbox deterministic equal-score ordering remains qualification-sensitive");

    OwnedRawOutputs owner;
    constexpr float captured_score = 0.4140625F;
    owner.set_candidate(8, 84, 76, 1, raw_logit_with_exact_sigmoid(captured_score),
                        {5.0F, 4.5F, 5.0F, 4.0F});
    const auto result = trtmc::decode_sam2_bbox_outputs(owner.views(), 1280, 1088);
    check(result.original_height == 1280 && result.original_width == 1088,
          "sam2 bbox retains original image geometry");
    check(result.detections.size() == 1, "sam2 bbox synthetic capture has one detection");
    const auto& detection = trtmc::require_exactly_one_sam2_bbox_detection(result);
    check(detection.label == 1 && detection.flattened_anchor_index == 84U * 128U + 76U,
          "sam2 bbox captured detection retains label and flattened anchor");
    check_equal(detection.score, captured_score,
                "sam2 bbox synthetic raw logit reproduces captured score exactly");
    check(detection.model_xyxy_1024 == std::array<float, 4>{572.0F, 640.0F, 652.0F, 708.0F},
          "sam2 bbox reproduces captured model-space bbox exactly");
    check(detection.original_xyxy == std::array<float, 4>{607.75F, 800.0F, 692.75F, 885.0F},
          "sam2 bbox applies captured anisotropic original-space scaling exactly once");
}

void test_bfloat16_input_half_stride_prior_and_no_clipping() {
    OwnedRawOutputs owner(trtmc::Sam2BBoxDataType::kBFloat16);
    owner.set_candidate(8, 0, 0, 1, 4.0F, {1.0F, 2.0F, 3.0F, 4.0F});
    const auto result = trtmc::decode_sam2_bbox_outputs(owner.views(), 1280, 1088);
    check(result.detections.size() == 1 && result.detections[0].label == 1,
          "sam2 bbox accepts a homogeneous native BF16 output set");
    const auto& detection = result.detections[0];
    check(detection.model_xyxy_1024 == std::array<float, 4>{-4.0F, -12.0F, 28.0F, 36.0F},
          "sam2 bbox decodes BF16 LTRB from the half-stride prior");
    check(detection.original_xyxy == std::array<float, 4>{-4.25F, -15.0F, 29.75F, 45.0F},
          "sam2 bbox preserves unclipped coordinates through one anisotropic scale");
}

void test_bfloat16_captured_native_raw_rounds_to_golden_box() {
    OwnedRawOutputs owner(trtmc::Sam2BBoxDataType::kBFloat16);
    owner.set_candidate(16, 42, 37, 1, -0.33984375F, {1.703125F, 2.40625F, 3.15625F, 1.6953125F});
    const auto result = trtmc::decode_sam2_bbox_outputs(owner.views(), 1280, 1088);
    check(result.detections.size() == 1, "sam2 bbox captured native raw maps retain one detection");
    const auto& detection = trtmc::require_exactly_one_sam2_bbox_detection(result);
    check(detection.flattened_anchor_index == 128U * 128U + 42U * 64U + 37U && detection.label == 1,
          "sam2 bbox captured native raw maps retain the reference winning anchor");
    check_equal(detection.score, 0.416015625F, "sam2 bbox rounds captured native sigmoid to BF16");
    check(detection.model_xyxy_1024 == std::array<float, 4>{572.0F, 640.0F, 652.0F, 708.0F},
          "sam2 bbox rounds captured native distance decode at source BF16 boundaries");
    check(detection.original_xyxy == std::array<float, 4>{607.75F, 800.0F, 692.75F, 885.0F},
          "sam2 bbox scales the source-rounded captured native box in FP32");
}

void test_strict_threshold_single_label_global_top_k_and_ties() {
    const float threshold_logit = raw_logit_with_exact_sigmoid(trtmc::kSam2BBoxScoreThreshold);
    float above_logit = threshold_logit;
    while (reference_sigmoid(above_logit) <= trtmc::kSam2BBoxScoreThreshold) {
        above_logit = std::nextafter(above_logit, std::numeric_limits<float>::infinity());
    }

    OwnedRawOutputs strict_owner;
    strict_owner.set_candidate(8, 0, 0, 0, threshold_logit, {0.1F, 0.1F, 0.1F, 0.1F});
    strict_owner.set_candidate(8, 0, 1, 1, above_logit, {0.1F, 0.1F, 0.1F, 0.1F});
    const auto strict = trtmc::decode_sam2_bbox_outputs(strict_owner.views(), 1024, 1024);
    check(strict.detections.size() == 1 && strict.detections[0].flattened_anchor_index == 1 &&
              strict.detections[0].label == 1,
          "sam2 bbox uses strict score greater-than and one class per anchor");

    OwnedRawOutputs tie_owner;
    tie_owner.set_candidate(8, 0, 0, 0, 4.0F, {0.1F, 0.1F, 0.1F, 0.1F});
    tie_owner.set_class_value(8, 0, 0, 1, 4.0F);
    tie_owner.set_candidate(8, 0, 2, 1, 4.0F, {0.1F, 0.1F, 0.1F, 0.1F});
    const auto ties = trtmc::decode_sam2_bbox_outputs(tie_owner.views(), 1024, 1024);
    check(ties.detections.size() == 2 && ties.detections[0].flattened_anchor_index == 0 &&
              ties.detections[0].label == 0 && ties.detections[1].flattened_anchor_index == 2,
          "sam2 bbox deterministically orders score ties by anchor and class ties by first class");

    OwnedRawOutputs top_k_owner;
    for (int32_t x = 0; x < 100; ++x) {
        top_k_owner.set_candidate(8, 0, x, 1, 3.0F - static_cast<float>(x) * 0.001F,
                                  {0.1F, 0.1F, 0.1F, 0.1F});
    }
    top_k_owner.set_candidate(32, 31, 31, 0, 4.0F, {0.1F, 0.1F, 0.1F, 0.1F});
    const auto top_k = trtmc::decode_sam2_bbox_outputs(top_k_owner.views(), 1024, 1024);
    check(top_k.detections.size() == 100, "sam2 bbox applies one global pre-NMS top one hundred");
    check(top_k.detections.front().flattened_anchor_index == 21503,
          "sam2 bbox globally orders a strongest stride-thirty-two candidate");
    for (const auto& detection : top_k.detections) {
        check(detection.flattened_anchor_index != 99,
              "sam2 bbox globally drops the weakest candidate across levels");
    }
}

void test_class_agnostic_nms_boundary_and_unapplied_max_ten() {
    OwnedRawOutputs nms_owner;
    nms_owner.set_candidate(8, 0, 0, 0, 5.0F, {0.5F, 0.5F, 1.0F, 0.75F});
    nms_owner.set_candidate(8, 0, 1, 1, 4.0F, {0.5F, 0.5F, 1.0F, 0.75F});
    nms_owner.set_candidate(8, 0, 2, 1, 3.0F, {1.5625F, 0.5F, -0.0625F, 0.75F});
    const auto nms = trtmc::decode_sam2_bbox_outputs(nms_owner.views(), 1024, 1024);
    check(nms.detections.size() == 2 && nms.detections[0].label == 0 &&
              nms.detections[1].label == 1 && nms.detections[1].flattened_anchor_index == 1,
          "sam2 bbox class-agnostic NMS retains IoU exactly point two and suppresses above it");

    OwnedRawOutputs max_owner;
    for (int32_t x = 0; x < 11; ++x) {
        max_owner.set_candidate(32, 0, x, x % 2, 4.0F - static_cast<float>(x) * 0.01F,
                                {0.1F, 0.1F, 0.1F, 0.1F});
    }
    const auto maximum = trtmc::decode_sam2_bbox_outputs(max_owner.views(), 1024, 1024);
    check(maximum.detections.size() == 11,
          "sam2 bbox deliberately does not apply configured max-per-image ten");
}

void test_exact_one_gate_rejects_zero_and_multiple() {
    OwnedRawOutputs empty_owner;
    const auto empty = trtmc::decode_sam2_bbox_outputs(empty_owner.views(), 1024, 1024);
    check_throws<trtmc::Sam2BBoxPostprocessError>(
        [&] { (void)trtmc::require_exactly_one_sam2_bbox_detection(empty); }, "received 0",
        "sam2 bbox exact-one gate rejects zero detections");

    OwnedRawOutputs multiple_owner;
    multiple_owner.set_candidate(32, 0, 0, 0, 4.0F, {0.1F, 0.1F, 0.1F, 0.1F});
    multiple_owner.set_candidate(32, 0, 1, 1, 3.0F, {0.1F, 0.1F, 0.1F, 0.1F});
    const auto multiple = trtmc::decode_sam2_bbox_outputs(multiple_owner.views(), 1024, 1024);
    check_throws<trtmc::Sam2BBoxPostprocessError>(
        [&] { (void)trtmc::require_exactly_one_sam2_bbox_detection(multiple); }, "received 2",
        "sam2 bbox exact-one gate rejects multiple detections without implicit top one");
}

void test_raw_abi_and_nonfinite_values_fail_closed() {
    {
        OwnedRawOutputs owner;
        auto outputs = owner.views();
        outputs.bbox_cls_stride_8.shape[0] = 2;
        check_throws<trtmc::Sam2BBoxAbiError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(outputs, 1024, 1024); }, "shape drifted",
            "sam2 bbox rejects a non-batch-one NCHW map");
    }
    {
        OwnedRawOutputs owner;
        auto outputs = owner.views();
        --outputs.bbox_reg_stride_16.element_count;
        check_throws<trtmc::Sam2BBoxAbiError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(outputs, 1024, 1024); }, "element count",
            "sam2 bbox rejects a truncated raw map");
    }
    {
        OwnedRawOutputs owner;
        auto outputs = owner.views();
        outputs.bbox_cls_stride_32.data = nullptr;
        check_throws<trtmc::Sam2BBoxAbiError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(outputs, 1024, 1024); }, "must not be null",
            "sam2 bbox rejects a null raw map");
    }
    {
        OwnedRawOutputs owner;
        auto outputs = owner.views();
        outputs.bbox_reg_stride_32.data_type = trtmc::Sam2BBoxDataType::kBFloat16;
        check_throws<trtmc::Sam2BBoxAbiError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(outputs, 1024, 1024); },
            "same declared precision", "sam2 bbox rejects mixed raw precisions");
    }
    {
        OwnedRawOutputs owner;
        auto outputs = owner.views();
        outputs.bbox_cls_stride_8.data_type = static_cast<trtmc::Sam2BBoxDataType>(99);
        check_throws<trtmc::Sam2BBoxAbiError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(outputs, 1024, 1024); },
            "unsupported data type", "sam2 bbox rejects an unknown raw precision");
    }
    {
        OwnedRawOutputs owner;
        owner.set_class_value(16, 3, 4, 0, std::numeric_limits<float>::quiet_NaN());
        check_throws<trtmc::Sam2BBoxAbiError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(owner.views(), 1024, 1024); },
            "NaN or infinity", "sam2 bbox rejects nonfinite logits even when unselected");
    }
    {
        OwnedRawOutputs owner;
        owner.set_regression_value(32, 3, 1, 1, std::numeric_limits<float>::infinity());
        check_throws<trtmc::Sam2BBoxAbiError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(owner.views(), 1024, 1024); },
            "NaN or infinity", "sam2 bbox rejects nonfinite regression values");
    }
    {
        OwnedRawOutputs owner;
        check_throws<trtmc::Sam2BBoxAbiError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(owner.views(), 0, 1024); }, "positive",
            "sam2 bbox rejects invalid original image geometry");
    }
}

void test_arithmetic_overflow_and_invalid_boxes_fail_closed() {
    {
        OwnedRawOutputs owner;
        // Overflow must fail even though the anchor cannot pass its score.
        owner.set_regression_value(8, 0, 0, 0, std::numeric_limits<float>::max());
        check_throws<trtmc::Sam2BBoxPostprocessError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(owner.views(), 1024, 1024); },
            "distance scaling overflowed",
            "sam2 bbox rejects distance overflow before candidate selection");
    }
    {
        OwnedRawOutputs owner;
        owner.set_candidate(8, 0, 0, 0, 4.0F, {-1.0F, -1.0F, -1.0F, -1.0F});
        check_throws<trtmc::Sam2BBoxPostprocessError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(owner.views(), 1024, 1024); },
            "non-positive-area", "sam2 bbox rejects inverted selected boxes");
    }
    {
        OwnedRawOutputs owner;
        owner.set_candidate(8, 0, 0, 0, 4.0F, {2.0e18F, 2.0e18F, 2.0e18F, 2.0e18F});
        check_throws<trtmc::Sam2BBoxPostprocessError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(owner.views(), 1024, 1024); },
            "box-area arithmetic", "sam2 bbox rejects NMS area overflow");
    }
    {
        OwnedRawOutputs owner;
        owner.set_candidate(8, 0, 0, 0, 5.0F, {3.75e18F, 0.5F, -1.25e18F, 1.25e18F});
        owner.set_candidate(8, 0, 1, 1, 4.0F, {-1.25e18F, 0.5F, 3.75e18F, 1.25e18F});
        check_throws<trtmc::Sam2BBoxPostprocessError>(
            [&] { (void)trtmc::decode_sam2_bbox_outputs(owner.views(), 1024, 1024); },
            "overlap arithmetic", "sam2 bbox rejects NMS union overflow");
    }
    {
        OwnedRawOutputs owner;
        owner.set_candidate(8, 0, 0, 0, 4.0F, {-1.25e31F, 0.5F, 2.5e31F, 0.5F});
        check_throws<trtmc::Sam2BBoxPostprocessError>(
            [&] {
                (void)trtmc::decode_sam2_bbox_outputs(owner.views(), 1024,
                                                      std::numeric_limits<int32_t>::max());
            },
            "original-space rescaling overflowed",
            "sam2 bbox rejects original-space multiplication overflow");
    }
}

} // namespace

int main() {
    test_contract_and_captured_reference_box();
    test_bfloat16_input_half_stride_prior_and_no_clipping();
    test_bfloat16_captured_native_raw_rounds_to_golden_box();
    test_strict_threshold_single_label_global_top_k_and_ties();
    test_class_agnostic_nms_boundary_and_unapplied_max_ten();
    test_exact_one_gate_rejects_zero_and_multiple();
    test_raw_abi_and_nonfinite_values_fail_closed();
    test_arithmetic_overflow_and_invalid_boxes_fail_closed();
    std::cout << "SAM2 pure C++ exact bbox postprocess tests passed\n";
    return 0;
}
