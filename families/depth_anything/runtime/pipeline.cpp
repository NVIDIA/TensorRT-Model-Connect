/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/depth_anything/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace trtmc {
namespace {

const Tensor& require_output(const std::unordered_map<std::string, Tensor>& outputs,
                             const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error("depth_anything engine did not produce " + std::string(name));
    return found->second;
}

// Nearest-neighbour resample of the engine's square depth map back onto the
// caller's image. The family owns this because it resized the input itself.
std::vector<float> resample(const float* source, int32_t source_h, int32_t source_w,
                            int32_t target_h, int32_t target_w) {
    std::vector<float> output(static_cast<std::size_t>(target_h) * target_w);
    for (int32_t y = 0; y < target_h; ++y) {
        const int32_t src_y =
            std::min(source_h - 1, static_cast<int32_t>((static_cast<float>(y) + 0.5F) *
                                                        static_cast<float>(source_h) /
                                                        static_cast<float>(target_h)));
        for (int32_t x = 0; x < target_w; ++x) {
            const int32_t src_x =
                std::min(source_w - 1, static_cast<int32_t>((static_cast<float>(x) + 0.5F) *
                                                            static_cast<float>(source_w) /
                                                            static_cast<float>(target_w)));
            output[static_cast<std::size_t>(y) * target_w + x] =
                source[static_cast<std::size_t>(src_y) * source_w + src_x];
        }
    }
    return output;
}

} // namespace

DepthAnythingPipeline::DepthAnythingPipeline(std::unique_ptr<ITrtModule> model,
                                             DepthAnythingPreprocessConfig preprocess_config)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("DepthAnythingPipeline: invalid model");
}

MonocularDepthResult DepthAnythingPipeline::estimate_depth(const float* pixels, int32_t height,
                                                           int32_t width) {
    auto pixel_values = preprocess_depth_anything_image(pixels, height, width, preprocess_config_);

    Tensor image;
    image.data = pixel_values.data();
    image.shape = {1, 3, preprocess_config_.input_image_h, preprocess_config_.input_image_w};
    image.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", image}});
    const Tensor& depth = require_output(outputs, "predicted_depth");
    if (depth.shape.size() < 2)
        throw std::runtime_error("depth_anything engine output has unexpected rank");

    const auto source_h = static_cast<int32_t>(depth.shape[depth.shape.size() - 2]);
    const auto source_w = static_cast<int32_t>(depth.shape[depth.shape.size() - 1]);
    if (source_h <= 0 || source_w <= 0)
        throw std::runtime_error("depth_anything engine output has unexpected shape");

    MonocularDepthResult result;
    result.height = height;
    result.width = width;
    result.depth =
        resample(static_cast<const float*>(depth.data), source_h, source_w, height, width);
    return result;
}

} // namespace trtmc
