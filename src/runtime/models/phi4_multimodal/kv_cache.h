/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Phi4MultimodalKvCache: autoregressive KV cache state manager.
// HF equivalent: DynamicCache / past_key_values.
//
// Manages the full-capacity user-owned buffers consumed and updated in place
// by TensorRT IKVCacheUpdateLayer.

#include "runtime/models/phi4_multimodal/inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class ITrtModule;
using TrtModule = ITrtModule;

// Explicit tensor names for KV cache I/O binding.
// Per-layer vectors hold expanded names; scalar names are for single inputs.
struct Phi4MultimodalKvCacheNames {
    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    std::vector<std::string> present_k;
    std::vector<std::string> present_v;
    std::string cache_write_indices{"cache_write_indices"};
    std::string key_value_lengths{"key_value_lengths"};
    std::string position_id{"position_id"};
};

class Phi4MultimodalKvCache : public Phi4MultimodalInferenceState {
  public:
    // Allocate cache buffers for the given configuration.
    // kv_dim = num_kv_heads * head_dim (size of one K or V row per layer).
    // cache_dtype controls the element type for K/V cache buffers (default FP32).
    // names provides explicit tensor names for engine I/O binding.
    Phi4MultimodalKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                          cudaStream_t stream, DType cache_dtype = DType::kFloat16,
                          Phi4MultimodalKvCacheNames names = {});

    // --- Phi4MultimodalInferenceState overrides ---
    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return max_length_; }
    int32_t num_layers() const override { return num_layers_; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "dense_kv_cache"; }
    bool ok() const override;

    // --- Phi4MultimodalKvCache-specific methods (not on the interface) ---

    // Direct access for advanced use (cross-attention, VL embedding).
    DeviceTensor& cache_k(int32_t layer) { return cache_k_[static_cast<std::size_t>(layer)]; }
    DeviceTensor& cache_v(int32_t layer) { return cache_v_[static_cast<std::size_t>(layer)]; }

    // Confirm that one native prefill chunk updated the aliased cache in place,
    // then advance the logical length.
    void commit_prefill(const std::vector<const void*>& present_k,
                        const std::vector<const void*>& present_v, int32_t seq_len);

    // Bind the same cache storage to the split prefill engine.
    void bind_cache_inputs(TrtModule& module);

  private:
    void validate_contract(TrtModule& module) const;
    void bind_native_cache(TrtModule& module);
    void validate_aliases(const std::vector<const void*>& present_k,
                          const std::vector<const void*>& present_v) const;
    void write_position_input(TensorMap& inputs, int32_t seq_len);

    std::vector<DeviceTensor> cache_k_; // [num_layers], shape [max_length, kv_dim]
    std::vector<DeviceTensor> cache_v_; // [num_layers]
    int32_t num_layers_{0};
    int32_t max_length_{0};
    int32_t kv_dim_{0};
    int32_t position_{0};
    // Buffers owned by this object — Tensor.data in prepare_step() points here.
    std::vector<int32_t> pos_buf_vec_;
    int32_t cache_write_index_{0};
    int32_t key_value_length_{0};
    bool has_position_input_{false};
    DType cache_dtype_{DType::kFloat32};
    Phi4MultimodalKvCacheNames names_;
};

} // namespace trtmc
