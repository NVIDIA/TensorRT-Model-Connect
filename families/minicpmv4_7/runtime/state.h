/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <vector>

namespace trtmc::minicpmv4_7 {

struct StateGeometry {
    int32_t capacity{0};
    int32_t linear_layers{0};
    int32_t full_layers{0};
    int32_t conv_dim{0};
    int32_t conv_kernel{0};
    int32_t linear_heads{0};
    int32_t linear_head_dim{0};
    int32_t kv_width{0};
    DType cache_dtype{DType::kBFloat16};
};

// Per-sequence decoder state: one KV row per processed token for each full-attention layer and
// the causal-conv window plus FP32 recurrent matrix of each Gated DeltaNet layer. The engine
// reads the committed state and writes "present" buffers; advance() commits them.
class DecoderState {
  public:
    DecoderState(const StateGeometry& geometry, cudaStream_t stream);

    void bind(ITrtModule& module);
    void reset();
    // KV slot of the next token and the additive mask over [cache rows, current token].
    void prepare(TensorMap& inputs);
    void advance();
    int32_t position() const { return position_; }
    int32_t capacity() const { return geometry_.capacity; }

  private:
    StateGeometry geometry_;
    cudaStream_t stream_;
    std::vector<DeviceTensor> conv_, conv_present_, ssm_, ssm_present_;
    std::vector<DeviceTensor> cache_k_, cache_v_, present_k_, present_v_;
    std::vector<float> mask_;
    int32_t position_{0};
    int32_t position_input_{0};
};

} // namespace trtmc::minicpmv4_7
