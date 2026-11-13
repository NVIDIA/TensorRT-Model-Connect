/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/vision_encoder.h"

#include "families/minicpmv4_6/runtime/engine_io.h"

#include <cstring>
#include <stdexcept>

namespace trtmc::minicpmv4_6 {

VisionEncoder::VisionEncoder(std::unique_ptr<ITrtModule> module, std::int32_t hidden_size,
                             DType dtype, std::int32_t patch_width, std::int32_t max_patches)
    : module_(std::move(module)), hidden_size_(hidden_size), dtype_(dtype),
      patch_width_(patch_width) {
    if (!module_ || !module_->ok())
        throw std::runtime_error("Invalid MiniCPM-V 4.6 vision engine");
    validate_vision_engine(*module_, dtype_, patch_width_, hidden_size_, max_patches);
}

std::vector<std::uint8_t> VisionEncoder::encode(const VisionSlice& slice) {
    const auto patches = static_cast<int64_t>(slice.height) * slice.width;
    if (patches <= 0 || slice.patches.size() != static_cast<std::size_t>(patches) * patch_width_ ||
        slice.position_ids.size() != static_cast<std::size_t>(patches) ||
        slice.window_index.size() != static_cast<std::size_t>(patches) ||
        slice.merge_index.size() * 4 != static_cast<std::size_t>(patches))
        throw std::invalid_argument("MiniCPM-V 4.6 vision slice inputs are inconsistent");
    TensorMap inputs{
        {"patches",
         {const_cast<float*>(slice.patches.data()), {patches, patch_width_}, DType::kFloat32}},
        {"position_ids",
         {const_cast<std::int32_t*>(slice.position_ids.data()), {patches}, DType::kInt32}},
        {"window_index",
         {const_cast<std::int32_t*>(slice.window_index.data()), {patches}, DType::kInt32}},
        {"merge_index",
         {const_cast<std::int32_t*>(slice.merge_index.data()), {patches / 4}, DType::kInt32}},
    };
    TensorMap outputs;
    try {
        outputs = module_->forward(inputs);
    } catch (...) {
        try {
            module_->sync();
        } catch (...) {
        }
        throw;
    }
    const auto& features = outputs.at("visual_features");
    if (features.dtype != dtype_ ||
        features.shape != std::vector<int64_t>{slice.tokens(), hidden_size_})
        throw std::runtime_error("MiniCPM-V 4.6 vision output has invalid geometry or dtype");
    std::vector<std::uint8_t> result(features.nbytes());
    std::memcpy(result.data(), features.data, result.size());
    return result;
}

} // namespace trtmc::minicpmv4_6
