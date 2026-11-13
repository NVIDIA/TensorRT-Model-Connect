/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/internal/scores.h"
#include "trtmc/task.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace trtmc::internal {

// Points are row-major [num_points, input_dim] float32 coordinates. The family
// owns any required normalization or augmentation and must reject unsupported
// input dimensions. There is no persistent state; one call segments one cloud.
struct PointsToSemanticSegmentationRequest {
    Span<const float> points;
    std::uint32_t num_points{0};
    std::uint32_t input_dim{3};
};

struct PointsToSemanticSegmentationResult {
    // Per-point class labels in the input point order.
    std::vector<std::int32_t> labels;
    std::uint32_t num_points{0};
    std::uint32_t num_classes{0};
    std::vector<std::int32_t> class_ids;
    std::vector<std::string> class_names; // Empty or one name per class ID.
    // Empty means unknown; class IDs remain local to this model output.
    std::string vocabulary_id;
    std::optional<std::int32_t> ignore_label;
    std::optional<std::int32_t> background_label;
    std::vector<float> class_scores; // Optional [num_points, num_classes].
    ScoreKind score_kind{ScoreKind::Logit};
};

class IPointsToSemanticSegmentation {
  public:
    using TaskInterface = IPointsToSemanticSegmentation;
    static constexpr std::string_view kTask = "points_to_semantic_segmentation";
    virtual ~IPointsToSemanticSegmentation() = default;
    virtual PointsToSemanticSegmentationResult
    run(const PointsToSemanticSegmentationRequest& request, ConfigView config) = 0;
};

} // namespace trtmc::internal
