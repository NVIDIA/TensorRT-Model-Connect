/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace trtmc::sam2_hoi {

inline constexpr std::size_t kHoiQueryCount = 1500;
inline constexpr std::size_t kHoiClassCount = 4;
inline constexpr std::size_t kHoiEmbeddingSize = 256;
inline constexpr std::size_t kHoiTopK = 300;

struct HoiDetection {
    std::array<float, 4> box_xyxy{};
    float score{0.0F};
    int32_t label{0};
    int32_t query_index{0};
};

struct HoiInteractionPair {
    int32_t source_detection_index{0};
    int32_t target_detection_index{0};
};

struct HoiPostprocessResult {
    std::vector<HoiDetection> detections;
    std::vector<HoiInteractionPair> interaction_pairs;
    std::vector<float> interaction_probabilities;
};

enum class HoiPostprocessStatus {
    kOk = 0,
    kInputSizeMismatch = 1,
    kNonFiniteValue = 2,
    kMissingInteractionEvaluator = 3,
    kInteractionProbabilityCountMismatch = 4,
};

// pair_embeddings contains pair_count consecutive 512-value rows. Each row is
// [source query embedding, target query embedding]. Pair rows use the same deterministic
// Cartesian-product order documented on postprocess_hoi().
using HoiInteractionProbabilityCallback = std::function<std::vector<float>(
    const std::vector<float>& pair_embeddings, std::size_t pair_count)>;

// Inputs are row-major [1500, 4] class probabilities, [1500, 4] normalized cxcywh boxes,
// and [1500, 256] query embeddings. Ordering is deterministic: flattened scores are sorted
// descending while ties retain the lower flattened index; NMS visits that order; the global
// NMS keep set is intersected by walking the class-aware keep list, so it cannot reorder it.
// Interaction candidates are all hand->label-2 pairs followed by all label-2->label-3 pairs,
// with each Cartesian product's left index as the outer loop. The callback is not invoked when
// no pair exists. A frame with no surviving hand returns an entirely empty result.
HoiPostprocessStatus
postprocess_hoi(const std::vector<float>& class_probabilities,
                const std::vector<float>& normalized_boxes_cxcywh,
                const std::vector<float>& query_embeddings,
                const HoiInteractionProbabilityCallback& interaction_probability_callback,
                HoiPostprocessResult& result);

// The supplied probabilities correspond one-to-one to the full deterministic candidate-pair
// sequence before the strict interaction-probability threshold is applied.
HoiPostprocessStatus postprocess_hoi_with_probabilities(
    const std::vector<float>& class_probabilities,
    const std::vector<float>& normalized_boxes_cxcywh, const std::vector<float>& query_embeddings,
    const std::vector<float>& interaction_probabilities, HoiPostprocessResult& result);

} // namespace trtmc::sam2_hoi
