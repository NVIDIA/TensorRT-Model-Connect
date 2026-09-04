/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/device_tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

namespace trtmc {

class ITrtModule;

// Fixed-capacity causal KV cache for the three Qwen3-Omni decoders.
class Qwen3OmniKvCache {
  public:
    Qwen3OmniKvCache(std::int32_t num_layers, std::int32_t max_length, std::int32_t kv_dim,
                     cudaStream_t stream, DType dtype);

    void reset();
    void bind_to(ITrtModule& module);
    void bind_cache_inputs(ITrtModule& module);
    void prepare_step(TensorMap& inputs, std::int32_t sequence_length = 1);
    void write_prefill_kv(const std::vector<const void*>& keys,
                          const std::vector<const void*>& values, std::int32_t sequence_length);
    void advance(std::int32_t tokens = 1);

    std::int32_t position() const { return position_; }
    std::int32_t max_length() const { return max_length_; }
    std::int32_t num_layers() const { return num_layers_; }
    std::size_t device_memory_bytes() const;
    bool ok() const;

  private:
    void validate_cache_inputs(ITrtModule& module) const;
    void write_positions(TensorMap& inputs, std::int32_t sequence_length);
    void write_mask(TensorMap& inputs, std::int32_t sequence_length);

    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    std::vector<DeviceTensor> present_k_;
    std::vector<DeviceTensor> present_v_;
    std::vector<float> mask_;
    std::vector<std::int32_t> positions_;
    std::int32_t num_layers_{0};
    std::int32_t max_length_{0};
    std::int32_t kv_dim_{0};
    std::int32_t position_{0};
    cudaStream_t stream_{nullptr};
    DType dtype_{DType::kFloat32};
    std::size_t element_size_{0};
};

} // namespace trtmc
