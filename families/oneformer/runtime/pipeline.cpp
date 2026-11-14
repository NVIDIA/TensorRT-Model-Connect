/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/oneformer/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::oneformer {
namespace {

constexpr const char* kInput = "pixel_values";
constexpr const char* kOutput = "semantic_scores";
constexpr std::uint32_t kChannels = 3;

// The official processor resizes 8-bit images; float pixels in [0, 1] are
// quantized to the same 8-bit values first.
std::vector<std::uint8_t> to_uint8(const internal::ImageView& image) {
    constexpr auto kInt32Max = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
    if (image.data == nullptr || image.channels != kChannels || image.height == 0 ||
        image.width == 0 || image.height > kInt32Max || image.width > kInt32Max)
        throw std::invalid_argument("OneFormer requires one non-empty RGB image");
    const auto count = static_cast<std::size_t>(image.height) * image.width * kChannels;
    if (count / kChannels / image.width != image.height)
        throw std::invalid_argument("OneFormer image dimensions overflow");
    std::vector<std::uint8_t> pixels(count);
    if (image.format == internal::ImageFormat::UInt8) {
        if (image.byte_size != count)
            throw std::invalid_argument("OneFormer uint8 RGB image has the wrong byte size");
        const auto* source = static_cast<const std::uint8_t*>(image.data);
        std::copy(source, source + count, pixels.begin());
    } else if (image.format == internal::ImageFormat::Float32) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            image.byte_size != count * sizeof(float))
            throw std::invalid_argument("OneFormer float32 RGB image has the wrong byte size");
        const auto* source = static_cast<const float*>(image.data);
        std::transform(source, source + count, pixels.begin(), [](float value) {
            if (!std::isfinite(value))
                throw std::invalid_argument("OneFormer float32 image contains a non-finite value");
            return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
        });
    } else {
        throw std::invalid_argument("OneFormer accepts uint8 or float32 RGB images");
    }
    return pixels;
}

ImageSize profile_size(const ITrtModule& engine, ProfileShapeSelector selector) {
    const auto shape = engine.input_profile_shape(kInput, 0, selector);
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0)
        throw std::runtime_error("OneFormer engine has no image profile");
    return {static_cast<std::int32_t>(shape[2]), static_cast<std::int32_t>(shape[3])};
}

} // namespace

SemanticSegmentation::SemanticSegmentation(std::unique_ptr<ITrtModule> engine, RuntimeConfig config)
    : engine_(std::move(engine)), config_(std::move(config)) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("OneFormer engine failed to load");
    if (!engine_->has_input(kInput) || !engine_->has_output(kOutput))
        throw std::runtime_error(
            "OneFormer engine does not expose pixel_values -> semantic_scores");
    if (config_.class_names.empty())
        throw std::runtime_error("OneFormer runtime config names no classes");
    min_input_ = profile_size(*engine_, ProfileShapeSelector::kMin);
    max_input_ = profile_size(*engine_, ProfileShapeSelector::kMax);
}

internal::SemanticSegmentationResult
SemanticSegmentation::run(const internal::ImageToSemanticSegmentationRequest& request,
                          internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("OneFormer has no runtime configuration");
    const auto& image = request.image;
    const auto pixels = to_uint8(image);
    const ImageSize source{static_cast<std::int32_t>(image.height),
                           static_cast<std::int32_t>(image.width)};
    const auto target = processed_size(source, config_.shortest_edge, config_.longest_edge);
    if (target.height < min_input_.height || target.width < min_input_.width ||
        target.height > max_input_.height || target.width > max_input_.width)
        throw std::invalid_argument(
            "OneFormer resizes this image to " + std::to_string(target.height) + "x" +
            std::to_string(target.width) + ", outside the engine's " +
            std::to_string(min_input_.height) + "x" + std::to_string(min_input_.width) + " to " +
            std::to_string(max_input_.height) + "x" + std::to_string(max_input_.width) +
            " image range");
    const auto resized = resize_uint8_antialias(pixels.data(), source, target, kChannels);
    auto planar = normalize_planar(resized, target, config_.normalization);

    Tensor input;
    input.data = planar.data();
    input.shape = {1, kChannels, target.height, target.width};
    input.dtype = DType::kFloat32;
    const auto outputs = engine_->forward({{kInput, input}});
    const auto found = outputs.find(kOutput);
    if (found == outputs.end())
        throw std::runtime_error("OneFormer engine did not return semantic_scores");
    const auto& scores = found->second;
    const auto classes = static_cast<std::int64_t>(config_.class_names.size());
    if (scores.dtype != DType::kFloat32 || scores.shape.size() != 4 || scores.shape[0] != 1 ||
        scores.shape[1] != classes || scores.shape[2] <= 0 || scores.shape[3] <= 0 ||
        scores.data == nullptr)
        throw std::runtime_error("OneFormer semantic_scores has an unexpected shape");
    const auto* values = static_cast<const float*>(scores.data);
    if (!std::all_of(values, values + scores.numel(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("OneFormer semantic_scores contains a non-finite value");

    internal::SemanticSegmentationResult result;
    result.labels = resize_argmax(
        values, static_cast<std::int32_t>(classes),
        {static_cast<std::int32_t>(scores.shape[2]), static_cast<std::int32_t>(scores.shape[3])},
        source);
    result.height = image.height;
    result.width = image.width;
    result.class_ids.resize(static_cast<std::size_t>(classes));
    std::iota(result.class_ids.begin(), result.class_ids.end(), 0);
    result.class_names = config_.class_names;
    return result;
}

} // namespace trtmc::oneformer
