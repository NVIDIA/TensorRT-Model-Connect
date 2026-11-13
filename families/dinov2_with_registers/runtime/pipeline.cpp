/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/dinov2_with_registers/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

const Tensor& require_output(const TensorMap& outputs, const char* name,
                             std::size_t expected_numel) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error(std::string("dinov2_with_registers engine did not return ") +
                                 name);
    const auto& tensor = found->second;
    if (tensor.data == nullptr || tensor.dtype != DType::kFloat32 ||
        tensor.numel() != expected_numel) {
        throw std::runtime_error(std::string("dinov2_with_registers engine returned an unusable ") +
                                 name);
    }
    return tensor;
}

} // namespace

Dinov2WithRegistersPipeline::Dinov2WithRegistersPipeline(std::unique_ptr<ITrtModule> engine,
                                                         Dinov2WithRegistersRuntimeConfig config,
                                                         std::string primary_task)
    : engine_(std::move(engine)), config_(std::move(config)),
      primary_task_(std::move(primary_task)) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("Dinov2WithRegistersPipeline: invalid model");
    if (config_.num_register_tokens <= 0 || config_.patch_size <= 0 || config_.grid <= 0 ||
        config_.hidden_size <= 0)
        throw std::runtime_error("dinov2_with_registers runtime config has invalid dimensions");
    if (config_.grid * config_.patch_size != config_.preprocess.input_image_h) {
        throw std::runtime_error(
            "dinov2_with_registers grid and patch size do not cover the input image");
    }
    if (config_.resample != 3)
        throw std::runtime_error("dinov2_with_registers is qualified for bicubic resize only");
}

std::vector<internal::TaskInstance> Dinov2WithRegistersPipeline::task_bindings() {
    return {internal::bind<internal::IImageToTokenAndPooledFeatures>(*this)};
}

internal::ImageTokenAndPooledFeaturesResult
Dinov2WithRegistersPipeline::run(const internal::ImageToTokenAndPooledFeaturesRequest& request,
                                 internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("dinov2_with_registers has no runtime configuration");
    const auto& image = request.image;
    if (image.format != internal::ImageFormat::Float32 || image.channels != 3 ||
        image.data == nullptr || image.height == 0 || image.width == 0 ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        image.width > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        static_cast<std::uint64_t>(image.height) >
            std::numeric_limits<std::size_t>::max() / image.width / 3 / sizeof(float) ||
        image.byte_size !=
            static_cast<std::size_t>(image.height) * image.width * 3 * sizeof(float)) {
        throw std::invalid_argument("dinov2_with_registers requires contiguous float32 RGB input");
    }

    const auto source_h = static_cast<std::int32_t>(image.height);
    const auto source_w = static_cast<std::int32_t>(image.width);
    auto pixel_values = preprocess_dinov2_with_registers_image(
        static_cast<const float*>(image.data), source_h, source_w, config_.preprocess);
    Tensor pixels;
    pixels.data = pixel_values.data();
    pixels.shape = {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w};
    pixels.dtype = DType::kFloat32;

    const auto grid = static_cast<std::uint64_t>(config_.grid);
    const auto width = static_cast<std::uint64_t>(config_.hidden_size);
    const auto rows = grid * grid + 1U + static_cast<std::uint64_t>(config_.num_register_tokens);
    auto outputs = engine_->forward({{"pixel_values", pixels}});
    const auto& hidden =
        require_output(outputs, "last_hidden_state", static_cast<std::size_t>(rows * width));
    const auto* values = static_cast<const float*>(hidden.data);

    internal::ImageTokenAndPooledFeaturesResult result;
    result.tokens.features.values.assign(values, values + rows * width);
    result.tokens.features.rows = rows;
    result.tokens.features.columns = width;
    result.tokens.grid_rows = grid;
    result.tokens.grid_columns = grid;

    // Class, register, then spatial patch rows retain the checkpoint order.
    result.tokens.tokens.reserve(static_cast<std::size_t>(rows));
    internal::ImageFeatureToken class_token;
    class_token.role = internal::ImageFeatureTokenRole::Class;
    result.tokens.tokens.push_back(class_token);
    for (std::int32_t index = 0; index < config_.num_register_tokens; ++index) {
        internal::ImageFeatureToken token;
        token.role = internal::ImageFeatureTokenRole::Register;
        result.tokens.tokens.push_back(token);
    }

    // Each patch's footprint is reported in ORIGINAL-image coordinates, so
    // the resize and the crop are both undone here. The transform comes from
    // the preprocessing seam rather than being re-derived, so the forward and
    // inverse rules cannot drift apart.
    const auto transform =
        compute_dinov2_with_registers_source_transform(source_h, source_w, config_.preprocess);
    const double patch = config_.patch_size;
    for (std::uint64_t row = 0; row < grid; ++row) {
        for (std::uint64_t column = 0; column < grid; ++column) {
            internal::ImageFeatureToken token;
            token.role = internal::ImageFeatureTokenRole::Patch;
            token.grid_row = row;
            token.grid_column = column;
            const double left =
                (static_cast<double>(column) * patch - transform.offset_x) / transform.scale_x;
            const double right =
                ((static_cast<double>(column) + 1.0) * patch - transform.offset_x) /
                transform.scale_x;
            const double top =
                (static_cast<double>(row) * patch - transform.offset_y) / transform.scale_y;
            const double bottom =
                ((static_cast<double>(row) + 1.0) * patch - transform.offset_y) / transform.scale_y;
            token.x_min = static_cast<float>(left / source_w);
            token.x_max = static_cast<float>(right / source_w);
            token.y_min = static_cast<float>(top / source_h);
            token.y_max = static_cast<float>(bottom / source_h);
            result.tokens.tokens.push_back(token);
        }
    }

    result.pooled.values.assign(values, values + width);
    // The reference's pooler is the class row of the normalised output, not a
    // mean over the patches and not a separate head.
    result.pooled.pooling = "class_token";
    // DINOv2 with registers's features are not unit length and the reference does not make
    // them so. Saying "none" keeps that visible instead of normalising here
    // and reporting a vector the reference never produced.
    result.pooled.normalization = "none";
    return result;
}

} // namespace trtmc
