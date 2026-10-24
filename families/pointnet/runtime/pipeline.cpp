/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/pointnet/runtime/pipeline.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace trtmc {

PointNetPipeline::PointNetPipeline(std::unique_ptr<ITrtModule> module, std::int32_t max_points,
                                   std::int32_t num_classes, std::int32_t input_dim)
    : module_(std::move(module)), max_points_(max_points), num_classes_(num_classes),
      input_dim_(input_dim) {
    if (module_ == nullptr || !module_->ok())
        throw std::runtime_error("PointNet: invalid engine module");
    if (max_points_ <= 0 || num_classes_ <= 0 || input_dim_ <= 0)
        throw std::runtime_error("PointNet: invalid runtime.json dimensions");
    if (!module_->has_input("point") || !module_->has_output("pred"))
        throw std::runtime_error("PointNet: engine tensor contract mismatch");
}

internal::PointsToSemanticSegmentationResult
PointNetPipeline::run(const internal::PointsToSemanticSegmentationRequest& request,
                      internal::ConfigView config) {
    static_cast<void>(config);
    if (request.points.data() == nullptr && request.num_points != 0)
        throw std::invalid_argument("PointNet: points pointer must not be null");
    if (request.num_points == 0 || request.num_points > static_cast<std::uint32_t>(max_points_))
        throw std::invalid_argument("PointNet: point count exceeds the bundle profile");
    if (request.input_dim != static_cast<std::uint32_t>(input_dim_))
        throw std::invalid_argument("PointNet: input dimension does not match the bundle");

    const auto point_count = static_cast<std::size_t>(request.num_points);
    const auto channel_count = static_cast<std::size_t>(input_dim_);
    std::vector<float> transposed(point_count * channel_count);
    for (std::size_t point = 0; point < point_count; ++point)
        for (std::size_t channel = 0; channel < channel_count; ++channel)
            transposed[channel * point_count + point] =
                request.points[point * channel_count + channel];

    Tensor input{transposed.data(),
                 {1, input_dim_, static_cast<std::int32_t>(point_count)},
                 DType::kFloat32};
    const auto outputs = module_->forward({{"point", input}});
    const auto it = outputs.find("pred");
    if (it == outputs.end() || it->second.data == nullptr || it->second.dtype != DType::kFloat32 ||
        it->second.numel() != point_count * static_cast<std::size_t>(num_classes_))
        throw std::runtime_error("PointNet: engine returned invalid pred");

    const auto* logits = static_cast<const float*>(it->second.data);
    internal::PointsToSemanticSegmentationResult result;
    result.labels.resize(point_count);
    result.num_points = request.num_points;
    result.num_classes = static_cast<std::uint32_t>(num_classes_);
    result.class_scores.assign(logits,
                               logits + point_count * static_cast<std::size_t>(num_classes_));
    result.score_kind = internal::ScoreKind::Logit;
    for (std::size_t point = 0; point < point_count; ++point) {
        const float* row = logits + point * static_cast<std::size_t>(num_classes_);
        result.labels[point] = static_cast<std::int32_t>(
            std::distance(row, std::max_element(row, row + num_classes_)));
    }
    return result;
}

} // namespace trtmc
