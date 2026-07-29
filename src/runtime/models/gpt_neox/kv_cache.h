/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/gpt_neox/inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class ITrtModule;
using TrtModule = ITrtModule;

struct GptNeoxKvCacheNames {
    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    std::vector<std::string> present_k;
    std::vector<std::string> present_v;
    std::string cache_write_indices{"cache_write_indices"};
    std::string key_value_lengths{"key_value_lengths"};
    std::string position_id{"position_id"};
};

// Owns one full-capacity KV allocation shared by the native TensorRT prefill
// and decode engines. TensorRT updates the aliased cache in place; this class
// only binds storage and advances logical sequence metadata.
class GptNeoxKvCache : public GptNeoxInferenceState {
  public:
    GptNeoxKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
                   DType cache_dtype = DType::kFloat16, GptNeoxKvCacheNames names = {});

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;
    int32_t position() const override { return position_; }
    int32_t max_length() const override { return max_length_; }
    int32_t preferred_cache_rows() const override { return max_length_; }
    int32_t num_layers() const override { return num_layers_; }
    bool needs_attention_mask() const override { return false; }
    std::size_t device_memory_bytes() const override;
    const char* state_type() const override { return "native_kv_cache"; }
    bool ok() const override;

    DeviceTensor& cache_k(int32_t layer) { return cache_k_[static_cast<std::size_t>(layer)]; }
    DeviceTensor& cache_v(int32_t layer) { return cache_v_[static_cast<std::size_t>(layer)]; }

    void append_prefill_kv(const std::vector<const void*>& present_k,
                           const std::vector<const void*>& present_v, int32_t seq_len);
    void bind_cache_inputs(TrtModule& module);

  private:
    void validate_native_kv_contract(TrtModule& module) const;
    void validate_native_aliases(const std::vector<const void*>& present_k,
                                 const std::vector<const void*>& present_v) const;
    void bind_native_cache(TrtModule& module);
    void write_native_kv_inputs(TensorMap& inputs, int32_t seq_len);
    void write_position_input(TensorMap& inputs, int32_t seq_len);

    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    int32_t num_layers_{0};
    int32_t max_length_{0};
    int32_t kv_dim_{0};
    int32_t position_{0};
    std::vector<int32_t> position_ids_;
    int32_t cache_write_index_{0};
    int32_t key_value_length_{0};
    DType cache_dtype_{DType::kFloat16};
    GptNeoxKvCacheNames names_;
};

} // namespace trtmc
