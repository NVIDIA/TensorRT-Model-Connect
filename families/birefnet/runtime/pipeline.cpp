/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/birefnet/runtime/pipeline.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc {
namespace {

const Tensor& require_output(const std::unordered_map<std::string, Tensor>& outputs,
                             const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error("birefnet engine did not produce " + std::string(name));
    return found->second;
}

} // namespace

BiRefNetSegmentationPipeline::BiRefNetSegmentationPipeline(std::unique_ptr<ITrtModule> model,
                                                           BiRefNetRuntimeConfig config)
    : model_(std::move(model)), config_(std::move(config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("BiRefNetSegmentationPipeline: engine failed to load");
}

SegmentResult BiRefNetSegmentationPipeline::segment(const float* pixels, std::int32_t height,
                                                    std::int32_t width) {
    auto prepared = preprocess_birefnet_image(pixels, height, width, config_.preprocess);

    Tensor input;
    input.data = prepared.data();
    input.shape = {1, 3, config_.preprocess.input_image_h, config_.preprocess.input_image_w};
    input.dtype = DType::kFloat32;
    auto outputs = model_->forward({{"pixel_values", input}});
    const Tensor& logits = require_output(outputs, "logits");

    SegmentResult result;
    result.height = height;
    result.width = width;
    // The network runs on its own square; the mask is returned at the source
    // resolution, so the caller never sees the resize.
    result.mask =
        mask_from_logits(static_cast<const float*>(logits.data), config_.preprocess.input_image_h,
                         config_.preprocess.input_image_w, height, width, config_.mask_threshold);
    return result;
}

} // namespace trtmc
