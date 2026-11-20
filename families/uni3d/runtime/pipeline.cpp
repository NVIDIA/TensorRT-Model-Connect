/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/uni3d/runtime/pipeline.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {
namespace {

// The CLIP space Uni3D is trained against (upstream --clip-model).
constexpr const char* kEmbeddingSpace = "open_clip/EVA02-E-14-plus";

} // namespace

Uni3DPipeline::Uni3DPipeline(std::unique_ptr<ITrtModule> module, std::int32_t min_points,
                             std::int32_t max_points, std::int32_t input_dim,
                             std::int32_t embed_dim)
    : module_(std::move(module)), min_points_(min_points), max_points_(max_points),
      input_dim_(input_dim), embed_dim_(embed_dim) {
    if (module_ == nullptr || !module_->ok())
        throw std::runtime_error("Uni3D: invalid engine module");
    if (min_points_ <= 0 || max_points_ < min_points_ || input_dim_ <= 0 || embed_dim_ <= 0)
        throw std::runtime_error("Uni3D: invalid runtime.json dimensions");
    if (!module_->has_input("points") || !module_->has_output("embedding"))
        throw std::runtime_error("Uni3D: engine tensor contract mismatch");
}

internal::SemanticEmbeddingResult
Uni3DPipeline::run(const internal::PointsToEmbeddingRequest& request, internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("uni3d has no runtime configuration");
    if (request.input_dim != static_cast<std::uint32_t>(input_dim_))
        throw std::invalid_argument("Uni3D: points must have " + std::to_string(input_dim_) +
                                    " columns (xyz, rgb)");
    if (request.num_points < static_cast<std::uint32_t>(min_points_) ||
        request.num_points > static_cast<std::uint32_t>(max_points_))
        throw std::invalid_argument("Uni3D: point count must be between " +
                                    std::to_string(min_points_) + " and " +
                                    std::to_string(max_points_));
    const auto count = static_cast<std::size_t>(request.num_points);
    const auto columns = static_cast<std::size_t>(input_dim_);
    if (request.points.data() == nullptr || request.points.size() != count * columns)
        throw std::invalid_argument("Uni3D: points span does not match num_points * input_dim");

    // The engine reads [1, N, 6] row-major, the request layout. Host inputs are only copied
    // to the device, never written, so the caller's buffer is passed without a copy.
    Tensor input{const_cast<float*>(request.points.data()),
                 {1, static_cast<std::int32_t>(count), input_dim_},
                 DType::kFloat32};
    const auto outputs = module_->forward({{"points", input}});
    const auto it = outputs.find("embedding");
    if (it == outputs.end() || it->second.data == nullptr || it->second.dtype != DType::kFloat32 ||
        it->second.numel() != static_cast<std::size_t>(embed_dim_))
        throw std::runtime_error("Uni3D: engine returned an invalid embedding");

    const auto* raw = static_cast<const float*>(it->second.data);
    double squared = 0.0;
    for (std::int32_t i = 0; i < embed_dim_; ++i)
        squared += static_cast<double>(raw[i]) * raw[i];
    const double norm = std::sqrt(squared);
    if (!std::isfinite(norm) || norm == 0.0)
        throw std::runtime_error("Uni3D: engine returned a zero or non-finite embedding");

    internal::SemanticEmbeddingResult result;
    result.values.resize(static_cast<std::size_t>(embed_dim_));
    for (std::int32_t i = 0; i < embed_dim_; ++i)
        result.values[static_cast<std::size_t>(i)] = static_cast<float>(raw[i] / norm);
    result.embedding_space = kEmbeddingSpace;
    result.pooling = "class_token";
    result.normalization = "l2";
    return result;
}

} // namespace trtmc
