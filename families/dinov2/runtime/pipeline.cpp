/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

std::vector<uint8_t> rgb_bytes(const internal::ImageView& image) {
    if (image.data == nullptr || image.channels != 3 || image.height == 0 || image.width == 0 ||
        image.height > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        image.width > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        static_cast<uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float))
        throw std::invalid_argument("DINOv2 requires a non-empty contiguous RGB image");
    const auto count = static_cast<std::size_t>(image.height) * image.width * 3U;
    if (image.format == internal::ImageFormat::UInt8) {
        if (image.byte_size != count)
            throw std::invalid_argument("DINOv2 uint8 RGB image size does not match its shape");
        const auto* data = static_cast<const uint8_t*>(image.data);
        return {data, data + count};
    }
    if (image.format != internal::ImageFormat::Float32 || image.byte_size != count * sizeof(float))
        throw std::invalid_argument("DINOv2 float32 RGB image size does not match its shape");
    // The reference processor resamples 8-bit images; [0,1] floats map back exactly.
    const auto* data = static_cast<const float*>(image.data);
    std::vector<uint8_t> bytes(count);
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(data[index]))
            throw std::invalid_argument("DINOv2 float32 RGB image contains a non-finite value");
        // Exactly std::lround for this nonnegative range: the double sum is exact.
        const float scaled = std::clamp(data[index], 0.0F, 1.0F) * 255.0F;
        bytes[index] = static_cast<uint8_t>(static_cast<double>(scaled) + 0.5);
    }
    return bytes;
}

} // namespace

Dinov2FeaturePipeline::Dinov2FeaturePipeline(std::unique_ptr<ITrtModule> model,
                                             Dinov2RuntimeConfig config)
    : model_(std::move(model)), config_(std::move(config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("Dinov2FeaturePipeline: invalid model");
    const auto& preprocess = config_.preprocess;
    if (config_.patch_size <= 0 || config_.hidden_size <= 0 || config_.num_register_tokens < 0 ||
        preprocess.input_image_h % config_.patch_size != 0 ||
        preprocess.input_image_w % config_.patch_size != 0)
        throw std::runtime_error("DINOv2 runtime.json does not match its engine contract");
    grid_rows_ = static_cast<uint64_t>(preprocess.input_image_h / config_.patch_size);
    grid_columns_ = static_cast<uint64_t>(preprocess.input_image_w / config_.patch_size);
    token_count_ =
        1U + static_cast<uint64_t>(config_.num_register_tokens) + grid_rows_ * grid_columns_;
}

internal::ImageTokenAndPooledFeaturesResult
Dinov2FeaturePipeline::run(const internal::ImageToTokenAndPooledFeaturesRequest& request,
                           internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("DINOv2 has no runtime configuration");
    const auto& image = request.image;
    const auto bytes = rgb_bytes(image);
    const auto height = static_cast<int32_t>(image.height);
    const auto width = static_cast<int32_t>(image.width);
    const auto& preprocess = config_.preprocess;
    auto pixel_values = preprocess_dinov2_image(bytes.data(), height, width, preprocess);
    const auto geometry = compute_dinov2_image_geometry(height, width, preprocess);

    Tensor input{pixel_values.data(),
                 {1, 3, preprocess.input_image_h, preprocess.input_image_w},
                 DType::kFloat32};
    const auto outputs = model_->forward({{"pixel_values", input}});
    const auto output = outputs.find("last_hidden_state");
    if (output == outputs.end())
        throw std::runtime_error("DINOv2 engine did not return last_hidden_state");
    const auto& hidden = output->second;
    const auto columns = static_cast<uint64_t>(config_.hidden_size);
    if (hidden.data == nullptr || hidden.dtype != DType::kFloat32 ||
        hidden.shape != std::vector<int64_t>{1, static_cast<int64_t>(token_count_),
                                             static_cast<int64_t>(columns)})
        throw std::runtime_error("DINOv2 engine output does not match its token contract");

    internal::ImageTokenAndPooledFeaturesResult result;
    auto& tokens = result.tokens;
    // forward() returns a view of a staging buffer that the next call reuses; copy it now.
    const auto* values = static_cast<const float*>(hidden.data);
    tokens.features.values.assign(values, values + token_count_ * columns);
    tokens.features.rows = token_count_;
    tokens.features.columns = columns;
    tokens.grid_rows = grid_rows_;
    tokens.grid_columns = grid_columns_;
    tokens.tokens.reserve(token_count_);
    internal::ImageFeatureToken prefix;
    prefix.role = internal::ImageFeatureTokenRole::Class;
    tokens.tokens.push_back(prefix);
    prefix.role = internal::ImageFeatureTokenRole::Register;
    for (int32_t index = 0; index < config_.num_register_tokens; ++index)
        tokens.tokens.push_back(prefix);
    // Patch footprints in normalized source coordinates: the resize maps the whole
    // source onto the resized image, so normalize crop-space edges by its size.
    const auto patch = static_cast<float>(config_.patch_size);
    const auto resized_h = static_cast<float>(geometry.resized_h);
    const auto resized_w = static_cast<float>(geometry.resized_w);
    for (uint64_t row = 0; row < grid_rows_; ++row) {
        for (uint64_t column = 0; column < grid_columns_; ++column) {
            internal::ImageFeatureToken token;
            token.role = internal::ImageFeatureTokenRole::Patch;
            token.grid_row = row;
            token.grid_column = column;
            token.x_min = (static_cast<float>(geometry.crop_x) + patch * column) / resized_w;
            token.x_max = (static_cast<float>(geometry.crop_x) + patch * (column + 1)) / resized_w;
            token.y_min = (static_cast<float>(geometry.crop_y) + patch * row) / resized_h;
            token.y_max = (static_cast<float>(geometry.crop_y) + patch * (row + 1)) / resized_h;
            tokens.tokens.push_back(token);
        }
    }
    // Dinov2Model's pooler_output is the final-layernorm CLS row.
    result.pooled.values.assign(tokens.features.values.begin(),
                                tokens.features.values.begin() +
                                    static_cast<std::ptrdiff_t>(columns));
    result.pooled.pooling = "cls";
    result.pooled.normalization = "none";
    return result;
}

} // namespace trtmc
