/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpm_ocr/runtime/vision_encoder.h"

#include <cstring>
#include <stdexcept>

namespace trtmc::minicpm_ocr {

VisionEncoder::VisionEncoder(std::unique_ptr<ITrtModule> engine, int32_t hidden_size, DType dtype,
                             int32_t patch_values, int32_t max_piece_patches, int32_t merge_size)
    : engine_(std::move(engine)), hidden_size_(hidden_size), dtype_(dtype),
      patch_values_(patch_values), merge_size_(merge_size) {
    if (!engine_ || !engine_->ok() || hidden_size_ <= 0 || patch_values_ <= 0 || merge_size_ <= 0 ||
        !engine_->has_input("patches") || !engine_->has_input("position_ids") ||
        !engine_->has_output("features") || engine_->tensor_dtype("patches") != DType::kFloat32 ||
        engine_->tensor_dtype("position_ids") != DType::kInt32 ||
        engine_->tensor_dtype("features") != dtype_)
        throw std::runtime_error("Invalid MiniCPM-OCR vision engine");
    const auto maximum =
        engine_->input_profile_shape("patches", engine_->profile_idx(), ProfileShapeSelector::kMax);
    if (maximum != std::vector<int64_t>{max_piece_patches, patch_values_})
        throw std::runtime_error("MiniCPM-OCR vision engine profile disagrees with runtime.json");
}

std::size_t VisionEncoder::encode(const ProcessedImage& image, std::vector<uint8_t>& features) {
    std::size_t appended = 0;
    const std::size_t row_bytes = static_cast<std::size_t>(hidden_size_) * dtype_size(dtype_);
    for (const auto& piece : image.pieces) {
        const int64_t count = static_cast<int64_t>(piece.height) * piece.width;
        const int64_t tokens = piece.tokens(merge_size_);
        if (count <= 0 || piece.patches.size() != static_cast<std::size_t>(count) * patch_values_ ||
            piece.position_ids.size() != static_cast<std::size_t>(count))
            throw std::invalid_argument("MiniCPM-OCR image piece has inconsistent geometry");
        TensorMap inputs;
        inputs["patches"] = Tensor{
            const_cast<float*>(piece.patches.data()), {count, patch_values_}, DType::kFloat32};
        inputs["position_ids"] =
            Tensor{const_cast<int32_t*>(piece.position_ids.data()), {count}, DType::kInt32};
        TensorMap outputs;
        try {
            outputs = engine_->forward(inputs);
        } catch (...) {
            try {
                engine_->sync();
            } catch (...) {
            }
            throw;
        }
        const auto& result = outputs.at("features");
        if (result.dtype != dtype_ || result.shape != std::vector<int64_t>{tokens, hidden_size_})
            throw std::runtime_error("MiniCPM-OCR vision output has invalid geometry or dtype");
        const auto offset = features.size();
        features.resize(offset + static_cast<std::size_t>(tokens) * row_bytes);
        std::memcpy(features.data() + offset, result.data,
                    static_cast<std::size_t>(tokens) * row_bytes);
        appended += static_cast<std::size_t>(tokens);
    }
    return appended;
}

} // namespace trtmc::minicpm_ocr
