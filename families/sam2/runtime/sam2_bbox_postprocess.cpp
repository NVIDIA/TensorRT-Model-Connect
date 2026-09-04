/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam2/runtime/sam2_bbox_postprocess.h"

#include "families/sam2/runtime/sam2_engine_contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

struct LevelView {
    int32_t stride;
    int32_t height;
    int32_t width;
    const char* cls_name;
    const char* reg_name;
    const Sam2BBoxTensorView* cls;
    const Sam2BBoxTensorView* reg;
    std::size_t anchor_offset;
};

struct Candidate {
    float score;
    int32_t label;
    std::size_t flattened_anchor_index;
    std::size_t level_index;
    std::size_t local_anchor_index;
};

struct DecodedCandidate {
    Candidate candidate;
    std::array<float, 4> box;
    float area;
};

float bfloat16_to_float(uint16_t value) {
    const uint32_t float_bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    static_assert(sizeof(result) == sizeof(float_bits), "SAM2 requires IEEE-754 float32");
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}

float round_to_bfloat16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t least_significant_retained_bit = (bits >> 16U) & 1U;
    bits += 0x7FFFU + least_significant_retained_bit;
    return bfloat16_to_float(static_cast<uint16_t>(bits >> 16U));
}

float load_value(const Sam2BBoxTensorView& tensor, std::size_t index) {
    return bfloat16_to_float(tensor.data[index]);
}

void validate_tensor(const Sam2BBoxTensorView& tensor, const char* name,
                     const std::array<int64_t, 4>& expected_shape) {
    if (tensor.shape != expected_shape) {
        throw Sam2BBoxAbiError(std::string("SAM2 bbox output ") + name + " NCHW shape drifted");
    }
    const auto expected_count =
        static_cast<std::size_t>(expected_shape[1] * expected_shape[2] * expected_shape[3]);
    if (tensor.element_count != expected_count) {
        throw Sam2BBoxAbiError(std::string("SAM2 bbox output ") + name +
                               " element count does not match its NCHW shape");
    }
    if (tensor.data == nullptr)
        throw Sam2BBoxAbiError(std::string("SAM2 bbox output ") + name + " must not be null");

    for (std::size_t index = 0; index < tensor.element_count; ++index) {
        if (!std::isfinite(load_value(tensor, index))) {
            throw Sam2BBoxAbiError(std::string("SAM2 bbox output ") + name +
                                   " contains NaN or infinity");
        }
    }
}

float sigmoid_float32(float value) {
    if (value >= 0.0F)
        return 1.0F / (1.0F + std::exp(-value));
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

std::array<LevelView, 3> make_levels(const Sam2BBoxRawOutputs& outputs) {
    constexpr std::size_t stride_8_anchors = 128U * 128U;
    constexpr std::size_t stride_16_anchors = 64U * 64U;
    return {{{8, 128, 128, "bbox_cls_stride_8", "bbox_reg_stride_8", &outputs.bbox_cls_stride_8,
              &outputs.bbox_reg_stride_8, 0},
             {16, 64, 64, "bbox_cls_stride_16", "bbox_reg_stride_16", &outputs.bbox_cls_stride_16,
              &outputs.bbox_reg_stride_16, stride_8_anchors},
             {32, 32, 32, "bbox_cls_stride_32", "bbox_reg_stride_32", &outputs.bbox_cls_stride_32,
              &outputs.bbox_reg_stride_32, stride_8_anchors + stride_16_anchors}}};
}

void validate_outputs(const std::array<LevelView, 3>& levels) {
    for (const auto& level : levels) {
        const std::array<int64_t, 4> cls_shape{1, 2, level.height, level.width};
        const std::array<int64_t, 4> reg_shape{1, 4, level.height, level.width};
        validate_tensor(*level.cls, level.cls_name, cls_shape);
        validate_tensor(*level.reg, level.reg_name, reg_shape);
        // Match the fail-closed reference: arithmetic overflow is checked for
        // every regression value, including anchors later removed by score.
        for (std::size_t index = 0; index < level.reg->element_count; ++index) {
            const float scaled = load_value(*level.reg, index) * static_cast<float>(level.stride);
            if (!std::isfinite(scaled)) {
                throw Sam2BBoxPostprocessError(std::string("SAM2 bbox stride-") +
                                               std::to_string(level.stride) +
                                               " distance scaling overflowed");
            }
        }
    }
}

std::vector<Candidate> select_candidates(const std::array<LevelView, 3>& levels) {
    std::vector<Candidate> candidates;
    for (std::size_t level_index = 0; level_index < levels.size(); ++level_index) {
        const auto& level = levels[level_index];
        const float score_threshold = round_to_bfloat16(kSam2BBoxScoreThreshold);
        const auto anchors =
            static_cast<std::size_t>(level.height) * static_cast<std::size_t>(level.width);
        for (std::size_t local_index = 0; local_index < anchors; ++local_index) {
            const float score_0 =
                round_to_bfloat16(sigmoid_float32(load_value(*level.cls, local_index)));
            const float score_1 =
                round_to_bfloat16(sigmoid_float32(load_value(*level.cls, anchors + local_index)));
            if (!std::isfinite(score_0) || !std::isfinite(score_1))
                throw Sam2BBoxPostprocessError("SAM2 bbox sigmoid produced NaN or infinity");

            // torch.max selects the first class on an exact tie.
            const int32_t label = score_1 > score_0 ? 1 : 0;
            const float score = label == 1 ? score_1 : score_0;
            if (score > score_threshold) {
                candidates.push_back(
                    {score, label, level.anchor_offset + local_index, level_index, local_index});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.score != rhs.score)
            return lhs.score > rhs.score;
        return lhs.flattened_anchor_index < rhs.flattened_anchor_index;
    });
    if (candidates.size() > kSam2BBoxPreNmsTopK)
        candidates.resize(kSam2BBoxPreNmsTopK);
    return candidates;
}

DecodedCandidate decode_candidate(const Candidate& candidate,
                                  const std::array<LevelView, 3>& levels) {
    const auto& level = levels[candidate.level_index];
    const auto anchors =
        static_cast<std::size_t>(level.height) * static_cast<std::size_t>(level.width);
    const auto x_index = candidate.local_anchor_index % static_cast<std::size_t>(level.width);
    const auto y_index = candidate.local_anchor_index / static_cast<std::size_t>(level.width);
    const float point_x =
        (static_cast<float>(x_index) + kSam2BBoxPointOffset) * static_cast<float>(level.stride);
    const float point_y =
        (static_cast<float>(y_index) + kSam2BBoxPointOffset) * static_cast<float>(level.stride);

    std::array<float, 4> distance{};
    for (std::size_t coordinate = 0; coordinate < distance.size(); ++coordinate) {
        distance[coordinate] = round_to_bfloat16(
            load_value(*level.reg, coordinate * anchors + candidate.local_anchor_index) *
            static_cast<float>(level.stride));
    }

    // The delivered source retains BF16 priors, strides, and raw maps through
    // distance decoding. Preserve those operation boundaries before retaining
    // the selected box as FP32 for native ABI and original-space scaling.
    std::array<float, 4> box{
        round_to_bfloat16(point_x - distance[0]), round_to_bfloat16(point_y - distance[1]),
        round_to_bfloat16(point_x + distance[2]), round_to_bfloat16(point_y + distance[3])};
    if (!std::all_of(box.begin(), box.end(), [](float value) { return std::isfinite(value); }))
        throw Sam2BBoxPostprocessError("SAM2 bbox decode produced NaN or infinity");
    if (box[2] <= box[0] || box[3] <= box[1]) {
        throw Sam2BBoxPostprocessError(
            "SAM2 bbox decode produced an invalid non-positive-area XYXY box");
    }

    const float width = box[2] - box[0];
    const float height = box[3] - box[1];
    const float area = width * height;
    if (!std::isfinite(width) || !std::isfinite(height) || !std::isfinite(area))
        throw Sam2BBoxPostprocessError("SAM2 NMS box-area arithmetic overflowed");
    return {candidate, box, area};
}

float checked_iou(const DecodedCandidate& lhs, const DecodedCandidate& rhs) {
    const float left = std::max(lhs.box[0], rhs.box[0]);
    const float top = std::max(lhs.box[1], rhs.box[1]);
    const float right = std::min(lhs.box[2], rhs.box[2]);
    const float bottom = std::min(lhs.box[3], rhs.box[3]);
    const float intersection_width = std::max(right - left, 0.0F);
    const float intersection_height = std::max(bottom - top, 0.0F);
    const float intersection = intersection_width * intersection_height;
    const float union_area = lhs.area + rhs.area - intersection;
    if (!std::isfinite(intersection_width) || !std::isfinite(intersection_height) ||
        !std::isfinite(intersection) || !std::isfinite(union_area)) {
        throw Sam2BBoxPostprocessError("SAM2 NMS overlap arithmetic overflowed");
    }
    if (union_area <= 0.0F)
        throw Sam2BBoxPostprocessError("SAM2 NMS produced an invalid IoU");
    const float iou = intersection / union_area;
    if (!std::isfinite(iou))
        throw Sam2BBoxPostprocessError("SAM2 NMS produced an invalid IoU");
    return iou;
}

std::vector<DecodedCandidate> class_agnostic_nms(std::vector<DecodedCandidate> decoded) {
    // select_candidates already established score-descending, anchor-ascending
    // order. Greedy NMS preserves that exact order.
    std::vector<DecodedCandidate> kept;
    kept.reserve(decoded.size());
    for (auto& candidate : decoded) {
        bool suppressed = false;
        for (const auto& selected : kept) {
            // IoU exactly 0.2 survives; only strictly greater overlap is
            // suppressed by the inspected source path.
            if (checked_iou(selected, candidate) > kSam2BBoxNmsIouThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed)
            kept.push_back(std::move(candidate));
    }
    return kept;
}

} // namespace

Sam2BBoxDetections decode_sam2_bbox_outputs(const Sam2BBoxRawOutputs& outputs) {
    const auto levels = make_levels(outputs);
    validate_outputs(levels);
    const auto candidates = select_candidates(levels);

    std::vector<DecodedCandidate> decoded;
    decoded.reserve(candidates.size());
    for (const auto& candidate : candidates)
        decoded.push_back(decode_candidate(candidate, levels));
    auto kept = class_agnostic_nms(std::move(decoded));

    // Convert through double before FP32 exactly as the reference constructs
    // its scale array. Scaling happens once, after NMS, with no clipping.
    const float scale_x = static_cast<float>(static_cast<double>(sam2::kOriginalImageWidth) /
                                             static_cast<double>(kSam2BBoxModelWidth));
    const float scale_y = static_cast<float>(static_cast<double>(sam2::kOriginalImageHeight) /
                                             static_cast<double>(kSam2BBoxModelHeight));
    if (!std::isfinite(scale_x) || !std::isfinite(scale_y))
        throw Sam2BBoxPostprocessError("SAM2 bbox original-space rescaling overflowed");

    Sam2BBoxDetections result;
    result.reserve(kept.size());
    for (const auto& item : kept) {
        Sam2BBoxDetection detection;
        detection.model_xyxy_1024 = item.box;
        detection.original_xyxy = {item.box[0] * scale_x, item.box[1] * scale_y,
                                   item.box[2] * scale_x, item.box[3] * scale_y};
        if (!std::all_of(detection.original_xyxy.begin(), detection.original_xyxy.end(),
                         [](float value) { return std::isfinite(value); })) {
            throw Sam2BBoxPostprocessError("SAM2 bbox original-space rescaling overflowed");
        }
        detection.score = item.candidate.score;
        detection.label = item.candidate.label;
        detection.flattened_anchor_index = item.candidate.flattened_anchor_index;
        result.push_back(detection);
    }
    return result;
}

const Sam2BBoxDetection&
require_exactly_one_sam2_bbox_detection(const Sam2BBoxDetections& detections) {
    if (detections.size() != 1) {
        throw Sam2BBoxPostprocessError(
            "SAM2 native v1 requires exactly one post-NMS detection; received " +
            std::to_string(detections.size()) + ", and no implicit top-1 policy is defined");
    }
    return detections.front();
}

} // namespace trtmc
