/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {
class ITrtModule;
}

namespace trtmc::minicpmv4_6 {

// Decoder geometry from runtime.json.
struct DecoderGeometry {
    std::int32_t attention_layers{0};
    std::int32_t deltanet_layers{0};
    std::int32_t kv_width{0};   // KV heads x head dimension
    std::int32_t conv_width{0}; // DeltaNet convolution channels
    std::int32_t conv_kernel{0};
    std::int32_t deltanet_heads{0};
    std::int32_t deltanet_head_dim{0};
    std::int32_t capacity{0}; // KV cache rows
    DType cache_dtype{DType::kFloat16};
};

// Per-request state of the hybrid decoder: the attention KV cache (rows
// [0, position) valid) and the FP32 DeltaNet convolution and recurrent states.
// Engines read the state through bound inputs and write present tensors; the
// state commits those after every launch on the engines' stream.
class DecoderState {
  public:
    DecoderState(const DecoderGeometry& geometry, cudaStream_t stream);

    void reset();
    std::int32_t position() const { return position_; }
    std::int32_t capacity() const { return geometry_.capacity; }

    // One-token decoder: caches, one-row present K/V and the recurrent state.
    void bind_decoder(ITrtModule& module);
    // Prefill engine: caches and recurrent state; its present K/V rows are read
    // from the engine after each launch.
    void bind_prefill(ITrtModule& module);

    // position_id and the additive attention mask for `rows` new tokens.
    void prepare(TensorMap& inputs, std::int32_t rows);
    void commit_decoder();
    void commit_prefill(const ITrtModule& module, std::int32_t rows);

  private:
    void bind_caches(ITrtModule& module);
    void bind_recurrent(ITrtModule& module);
    void commit_recurrent();

    DecoderGeometry geometry_;
    cudaStream_t stream_{nullptr};
    std::vector<DeviceTensor> cache_k_, cache_v_, present_k_, present_v_;
    std::vector<DeviceTensor> conv_, present_conv_, ssm_, present_ssm_;
    std::vector<float> mask_;
    std::vector<std::int32_t> positions_;
    std::int32_t position_{0};
};

} // namespace trtmc::minicpmv4_6
