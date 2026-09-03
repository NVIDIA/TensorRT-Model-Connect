/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/k2_horizon/inference_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct K2HorizonKvCacheNames {
    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    std::vector<std::string> present_k;
    std::vector<std::string> present_v;
    std::string cache_write_indices{"cache_write_indices"};
    std::string key_value_lengths{"key_value_lengths"};
    std::string position_id{"position_id"};
};

// Owns the fixed-capacity BF16 KV buffers used by the qualified K2-Horizon
// single-token TensorRT graph. The engine updates each cache in place: its
// present output must alias the corresponding cache input.
class K2HorizonKvCache : public K2HorizonInferenceState {
  public:
    K2HorizonKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim, cudaStream_t stream,
                     K2HorizonKvCacheNames names);

    void reset() override;
    void bind_to(TrtModule& module) override;
    void prepare_step(TensorMap& inputs, int32_t seq_len = 1) override;
    void advance(int32_t n_tokens = 1) override;

    int32_t position() const override { return position_; }
    int32_t max_length() const override { return max_length_; }
    std::size_t device_memory_bytes() const override;
    bool ok() const override;

  private:
    void validate_engine_contract(TrtModule& module) const;
    void bind_cache_aliases(TrtModule& module);

    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    int32_t num_layers_{0};
    int32_t max_length_{0};
    int32_t num_kv_heads_{0};
    int32_t position_{0};
    int32_t position_id_{0};
    int32_t cache_write_index_{0};
    int32_t key_value_length_{0};
    K2HorizonKvCacheNames names_;
    bool bound_{false};
};

} // namespace trtmc
