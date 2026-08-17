/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

inline constexpr int32_t kSam2BBoxModelHeight = 1024;
inline constexpr int32_t kSam2BBoxModelWidth = 1024;
inline constexpr float kSam2BBoxPointOffset = 0.5F;
inline constexpr float kSam2BBoxScoreThreshold = 0.35F;
inline constexpr std::size_t kSam2BBoxPreNmsTopK = 100;
inline constexpr float kSam2BBoxNmsIouThreshold = 0.2F;
inline constexpr std::size_t kSam2BBoxConfiguredMaxPerImage = 10;
inline constexpr bool kSam2BBoxAppliesConfiguredMaxPerImage = false;

// The inspected CUDA path does not specify equal-score ordering across
// devices. Native postprocessing uses flattened-anchor order to be
// deterministic, but that tie policy is deliberately not a parity claim.
inline constexpr bool kSam2BBoxTieOrderParityQualified = false;

enum class Sam2BBoxDataType : uint8_t {
    kFloat32 = 0,
    kBFloat16 = 1,
};

// A read-only, tightly packed host NCHW tensor. element_count is retained in
// addition to shape so callers cannot accidentally expose a truncated engine
// output buffer. BF16 elements use their native two-byte bit representation.
struct Sam2BBoxTensorView {
    const void* data{nullptr};
    Sam2BBoxDataType data_type{Sam2BBoxDataType::kFloat32};
    std::array<int64_t, 4> shape{};
    std::size_t element_count{0};
};

// Exact six-output ABI, in the same name and level order as the image engine.
struct Sam2BBoxRawOutputs {
    Sam2BBoxTensorView bbox_cls_stride_8;
    Sam2BBoxTensorView bbox_cls_stride_16;
    Sam2BBoxTensorView bbox_cls_stride_32;
    Sam2BBoxTensorView bbox_reg_stride_8;
    Sam2BBoxTensorView bbox_reg_stride_16;
    Sam2BBoxTensorView bbox_reg_stride_32;
};

struct Sam2BBoxDetection {
    std::array<float, 4> model_xyxy_1024{};
    std::array<float, 4> original_xyxy{};
    float score{0.0F};
    int32_t label{-1};
    std::size_t flattened_anchor_index{0};
};

struct Sam2BBoxDetections {
    std::vector<Sam2BBoxDetection> detections;
    int32_t original_height{0};
    int32_t original_width{0};
};

class Sam2BBoxPostprocessError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class Sam2BBoxAbiError final : public Sam2BBoxPostprocessError {
  public:
    using Sam2BBoxPostprocessError::Sam2BBoxPostprocessError;
};

// Decode the fixed batch-one 1024x1024 detector outputs. All input values are
// validated before selection. BF16 maps retain the source's BF16 sigmoid,
// threshold, stride, and point-arithmetic boundaries; selected results are
// then retained as FP32 for the native ABI and original-space scaling.
Sam2BBoxDetections decode_sam2_bbox_outputs(const Sam2BBoxRawOutputs& outputs,
                                            int32_t original_height, int32_t original_width);

// Native v1 has exactly one track. This gate rejects zero and multiple
// post-NMS detections instead of inventing an implicit top-one policy.
const Sam2BBoxDetection&
require_exactly_one_sam2_bbox_detection(const Sam2BBoxDetections& detections);

} // namespace trtmc
