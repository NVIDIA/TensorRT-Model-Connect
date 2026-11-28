/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "experts.h"

#include <string>
#include <unordered_map>

namespace trtmc::qwen3_8_flash_next {

struct SavedTensor {
    std::vector<char> bytes;
    std::vector<std::int64_t> shape;
    DType dtype;
    Tensor view() { return {bytes.data(), shape, dtype}; }
};

struct LayerOutput {
    SavedTensor hyper;
    std::unordered_map<std::string, SavedTensor> state;
};

// Input caches are never mutated. The caller may commit all returned layer
// states together only after success, or discard them on failure/cancellation.
LayerOutput run_decoder_layer(ITrtModule& layer, ITrtModule& expert, ITrtModule& combine,
                              const TensorMap& inputs,
                              const std::function<std::vector<char>(std::int32_t)>& load_expert);

} // namespace trtmc::qwen3_8_flash_next
