/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/videomae/runtime/pipeline.h"

#include "families/videomae/runtime/bundle_config.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace trtmc::videomae {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

const Tensor& require_predictions(const TensorMap& outputs, std::size_t tubes,
                                  const ClipGeometry& geometry) {
    const auto found = outputs.find("logits");
    if (found == outputs.end())
        throw std::runtime_error("videomae engine did not return logits");
    const auto& logits = found->second;
    const std::vector<std::int64_t> expected{1, static_cast<std::int64_t>(tubes),
                                             geometry.tube_values()};
    if (logits.data == nullptr || logits.dtype != DType::kFloat32 || logits.shape != expected)
        throw std::runtime_error("videomae engine returned logits of an unexpected shape or type");
    return logits;
}

} // namespace

MaskedVideoReconstruction::MaskedVideoReconstruction(std::unique_ptr<ITrtModule> engine,
                                                     ClipGeometry geometry,
                                                     PixelNormalization normalization)
    : engine_(std::move(engine)), geometry_(geometry), normalization_(normalization) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("videomae engine is not loaded");
    validate_geometry(geometry_);
    validate_engine(*engine_, geometry_);
}

internal::VideoResult
MaskedVideoReconstruction::run(const internal::MaskedVideoTextToVideoRequest& request,
                               internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("videomae has no runtime configuration");
    if (!request.prompt.empty())
        throw std::invalid_argument("videomae has no text condition; the prompt must be empty");
    const auto start = Clock::now();
    auto clip = source_clip(request.source.frames, geometry_);
    auto selection = select_tubes(request.mask, geometry_);
    auto values = pixel_values(clip, geometry_, normalization_);

    Tensor pixels;
    pixels.data = values.data();
    pixels.shape = {1, geometry_.frames, ClipGeometry::kChannels, geometry_.size, geometry_.size};
    pixels.dtype = DType::kFloat32;
    Tensor visible;
    visible.data = selection.visible.data();
    visible.shape = {static_cast<std::int64_t>(selection.visible.size())};
    visible.dtype = DType::kInt32;
    Tensor masked;
    masked.data = selection.masked.data();
    masked.shape = {static_cast<std::int64_t>(selection.masked.size())};
    masked.dtype = DType::kInt32;
    const auto prepared = Clock::now();
    const auto outputs = engine_->forward(
        {{"pixel_values", pixels}, {"visible_index", visible}, {"masked_index", masked}});
    const auto& predictions = require_predictions(outputs, selection.masked.size(), geometry_);
    reconstruct(clip, static_cast<const float*>(predictions.data), selection, geometry_);

    internal::VideoResult result;
    result.frames.pixels = std::move(clip);
    result.frames.height = static_cast<std::int32_t>(geometry_.size);
    result.frames.width = static_cast<std::int32_t>(geometry_.size);
    result.frames.channels = static_cast<std::int32_t>(ClipGeometry::kChannels);
    result.frames.num_frames = static_cast<std::int32_t>(geometry_.frames);
    // Frame i of the result is the reconstruction of source frame i.
    result.timestamps_seconds.assign(request.source.timestamps_seconds.begin(),
                                     request.source.timestamps_seconds.end());
    result.setup_ms = milliseconds(start, prepared);
    result.inference_ms = milliseconds(prepared, Clock::now());
    return result;
}

} // namespace trtmc::videomae
