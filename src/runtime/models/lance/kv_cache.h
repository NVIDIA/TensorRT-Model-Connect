/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Full-context user-owned Lance KV buffers consumed and updated in place by
// TensorRT IKVCacheUpdateLayer. The runtime mask covers only active KV rows;
// it preserves Lance's bidirectional vision span without scaling with capacity.

#include "runtime/models/lance/inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class ITrtModule;
using TrtModule = ITrtModule;

struct LanceKvCacheNames {
    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    std::vector<std::string> present_k;
    std::vector<std::string> present_v;
    std::string cache_write_indices{"cache_write_indices"};
    std::string key_value_lengths{"key_value_lengths"};
    std::string position_id{"position_id"};
    std::string attention_mask{"attention_mask"};
};

class LanceKvCache : public LanceInferenceState {
  public:
    LanceKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
                 DType cache_dtype = DType::kBFloat16, LanceKvCacheNames names = {});

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return max_length_; }
    int32_t preferred_cache_rows() const override { return max_length_; }
    int32_t num_layers() const override { return num_layers_; }
    bool needs_attention_mask() const override { return true; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "dense_kv_cache"; }
    bool ok() const override;

    DeviceTensor& cache_k(int32_t layer) { return cache_k_[static_cast<std::size_t>(layer)]; }
    DeviceTensor& cache_v(int32_t layer) { return cache_v_[static_cast<std::size_t>(layer)]; }

    // IKVCacheUpdate already wrote these rows into the aliased user buffer.
    // These methods validate the alias and advance only the logical length.
    void write_prefill_kv(const std::vector<const void*>& prefill_k,
                          const std::vector<const void*>& prefill_v, int32_t seq_len);
    void append_prefill_kv(const std::vector<const void*>& prefill_k,
                           const std::vector<const void*>& prefill_v, int32_t seq_len);

    // Preserve Lance's [causal, full vision span, causal] prompt semantics.
    // block_start/end are relative to this enqueue's token chunk.
    void prepare_prefill_with_bidirectional_block(TensorMap& inputs, int32_t seq_len,
                                                  int32_t block_start, int32_t block_end);

    void set_position(int32_t position);
    void bind_cache_inputs(TrtModule& module);

  private:
    void validate_contract(TrtModule& module) const;
    void bind_native_cache(TrtModule& module);
    void validate_aliases(const std::vector<const void*>& present_k,
                          const std::vector<const void*>& present_v) const;
    void write_position_inputs(TensorMap& inputs, int32_t seq_len);
    void write_native_scalars(TensorMap& inputs, int32_t seq_len);
    void write_causal_mask(TensorMap& inputs, int32_t seq_len);
    void write_segmented_mask(TensorMap& inputs, int32_t seq_len, int32_t block_start,
                              int32_t block_end);

    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    int32_t num_layers_{0};
    int32_t max_length_{0};
    int32_t kv_dim_{0};
    int32_t position_{0};
    std::vector<float> mask_buf_;
    std::vector<int32_t> pos_buf_vec_;
    std::vector<int32_t> mrope_pos_buf_;
    int32_t cache_write_index_{0};
    int32_t key_value_length_{0};
    bool has_position_input_{false};
    bool has_mrope_position_input_{false};
    bool has_attention_mask_input_{false};
    DType cache_dtype_{DType::kBFloat16};
    LanceKvCacheNames names_;
};

} // namespace trtmc
