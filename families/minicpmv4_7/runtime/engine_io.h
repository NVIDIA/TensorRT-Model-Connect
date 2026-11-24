/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace trtmc::minicpmv4_7 {

// An engine tensor the runtime binds by name, as the builder writes it; -1 marks a dimension that
// each call sets.
struct EngineTensor {
    const char* name;
    bool input;
    DType dtype;
    std::vector<int64_t> shape;
};

// An engine tensor as the loaded engine reports it.
struct ReportedTensor {
    bool present{false};
    bool input{false};
    DType dtype{DType::kFloat32};
    std::vector<int64_t> shape;
};

inline std::vector<EngineTensor> vision_io(DType work, int64_t patch_width, int64_t hidden) {
    return {{"pixel_patches", true, DType::kFloat32, {-1, patch_width}},
            {"position_ids", true, DType::kInt32, {-1}},
            {"window_index", true, DType::kInt32, {-1}},
            {"merge_index", true, DType::kInt32, {-1}},
            {"image_features", false, work, {-1, hidden}}};
}

// The decoder's per-token inputs and its logits, which the host copies four bytes per value;
// DecoderState::bind checks the state tensors and the attention mask.
inline std::vector<EngineTensor> decoder_io(DType work, int64_t hidden, int64_t vocab) {
    return {{"token_id", true, DType::kInt32, {1}},
            {"position_id", true, DType::kInt32, {1}},
            {"visual_embedding", true, work, {1, hidden}},
            {"use_visual_embedding", true, DType::kInt32, {1, 1}},
            {"rotary_position_offsets", true, DType::kInt32, {3}},
            {"logits", false, DType::kFloat32, {1, vocab}}};
}

// The first expected tensor that the engine lacks or reports in the other direction, with another
// dtype, rank or static dimension; empty when all match.
inline std::string first_mismatch(const std::vector<EngineTensor>& expected,
                                  const std::function<ReportedTensor(const char*)>& report) {
    for (const auto& tensor : expected) {
        const auto found = report(tensor.name);
        bool same = found.present && found.input == tensor.input && found.dtype == tensor.dtype &&
                    found.shape.size() == tensor.shape.size();
        for (std::size_t i = 0; same && i < tensor.shape.size(); ++i)
            same = tensor.shape[i] == -1 || found.shape[i] == tensor.shape[i];
        if (!same)
            return tensor.name;
    }
    return {};
}

} // namespace trtmc::minicpmv4_7
