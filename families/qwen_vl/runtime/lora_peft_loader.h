/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc {

// Owns normalized host buffers while exposing non-owning Tensor views for the
// existing GPU adapter cache upload path.
class QwenVlHostLoraAdapter {
  public:
    struct Buffer {
        std::vector<uint8_t> bytes;
        std::vector<int64_t> shape;
        DType dtype{DType::kFloat32};
    };

    std::unordered_map<std::string, Buffer> buffers;
    TensorMap tensor_views();
};

// Load a standard PEFT LoRA directory and normalize its tensors to the fixed
// shapes and dtypes declared by a LoRA-capable TensorRT engine.
QwenVlHostLoraAdapter qwen_vl_load_peft_lora_adapter(const std::string& adapter_dir,
                                                     const std::vector<TensorInfo>& engine_inputs);

} // namespace trtmc
