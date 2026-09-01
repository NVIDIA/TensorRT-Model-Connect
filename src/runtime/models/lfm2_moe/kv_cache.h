/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/lfm2_moe/inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct Lfm2MoeKvCacheNames {
    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    std::vector<std::string> present_k;
    std::vector<std::string> present_v;
    std::string cache_write_indices{"cache_write_indices"};
    std::string key_value_lengths{"key_value_lengths"};
    std::string position_id{"position_id"};
};

// Fixed-capacity cache for the attention layers in LFM2. The TensorRT
// IKVCacheUpdate outputs alias the cache inputs, so no per-step KV copy is
// required in C++.
class Lfm2MoeKvCache : public Lfm2MoeInferenceState {
  public:
    Lfm2MoeKvCache(int32_t num_layers, int32_t max_length, int32_t num_kv_heads, int32_t head_dim,
                   cudaStream_t stream, DType dtype, Lfm2MoeKvCacheNames names = {});

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;

    int32_t position() const override { return position_; }
    int32_t max_length() const override { return max_length_; }
    int32_t num_layers() const override { return num_layers_; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "lfm2_moe_native_kv_cache"; }
    bool ok() const override;

    DeviceTensor& cache_k(int32_t layer) { return cache_k_.at(static_cast<std::size_t>(layer)); }
    DeviceTensor& cache_v(int32_t layer) { return cache_v_.at(static_cast<std::size_t>(layer)); }

  private:
    void validate_contract(TrtModule& module) const;

    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    int32_t num_layers_{0};
    int32_t max_length_{0};
    int32_t num_kv_heads_{0};
    int32_t head_dim_{0};
    int32_t kv_dim_{0};
    int32_t position_{0};
    int32_t cache_write_index_{0};
    int32_t key_value_length_{0};
    std::vector<int32_t> positions_;
    cudaStream_t stream_{nullptr};
    DType dtype_{DType::kFloat16};
    Lfm2MoeKvCacheNames names_;
};

} // namespace trtmc
