/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/depth_anything_v2/runtime/pipeline.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

const Tensor& require_output(const TensorMap& outputs, const char* name) {
    const auto output = outputs.find(name);
    if (output == outputs.end()) {
        throw std::runtime_error(
            std::string("Depth Anything V2 engine did not return required output '") + name + "'");
    }
    return output->second;
}

} // namespace

DepthAnythingV2Pipeline::DepthAnythingV2Pipeline(std::unique_ptr<ITrtModule> model,
                                                 DepthAnythingV2PreprocessConfig preprocess_config)
    : model_(std::move(model)), preprocess_config_(std::move(preprocess_config)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("DepthAnythingV2Pipeline: invalid model");
}

GeometryResult DepthAnythingV2Pipeline::estimate_geometry(const float* pixels, int32_t height,
                                                          int32_t width) {
    auto pixel_values =
        preprocess_depth_anything_v2_image(pixels, height, width, preprocess_config_);
    const int32_t size = preprocess_config_.input_image_size;
    const std::vector<int64_t> input_shape{1, 3, size, size};
    Tensor input{pixel_values.data(), input_shape, DType::kFloat32};

    const auto outputs = model_->forward({{"pixel_values", input}});
    const auto& depth = require_output(outputs, "predicted_depth");
    if (depth.dtype != DType::kFloat32 || depth.shape != std::vector<int64_t>{1, size, size}) {
        throw std::runtime_error(
            "Depth Anything V2 output contract mismatch for 'predicted_depth'");
    }

    GeometryResult result;
    result.height = size;
    result.width = size;
    const auto area = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    result.depth.assign(static_cast<const float*>(depth.data),
                        static_cast<const float*>(depth.data) + area);

    // Depth Anything V2 predicts only a relative (uncalibrated) depth map: no
    // camera intrinsics and no 3D point cloud, unlike MoGe. `GeometryResult`
    // requires `points`/`mask` sized to the image regardless, so this fills
    // them with the values this family already uses to mean "no 3D
    // reconstruction here" (`points` at +infinity, `mask` clear) rather than
    // fabricate a plausible-looking point cloud this model never computed.
    // `depth`, `height`, and `width` above are the real, meaningful output.
    result.mask.assign(area, 0);
    result.points.assign(area * 3U, std::numeric_limits<float>::infinity());
    result.intrinsics = {};
    return result;
}

} // namespace trtmc
