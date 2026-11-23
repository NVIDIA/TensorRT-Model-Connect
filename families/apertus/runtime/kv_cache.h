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

namespace trtmc::apertus {

// Fixed-capacity K/V storage shared by the prefill and decode contexts of one engine.
// The engine updates it in place (TensorRT KV-cache update, present aliases cache);
// this class owns the device buffers and the logical sequence length.
class KvCache {
  public:
    KvCache(std::int32_t num_layers, std::int32_t num_kv_heads, std::int32_t head_dim,
            std::int32_t capacity, DType dtype, cudaStream_t stream);

    bool ok() const;
    std::int32_t capacity() const { return capacity_; }
    std::int32_t length() const { return length_; }

    // Validate the engine's cache contract and bind its cache inputs to this storage.
    void bind(ITrtModule& module) const;
    // Fill the cache-control inputs for appending `tokens` rows at the current length.
    void prepare(TensorMap& inputs, std::int32_t tokens);
    // Record that the engine has written the rows announced by prepare().
    void commit(std::int32_t tokens);
    void reset() { length_ = 0; }

    // Rows attention reads: the valid length rounded up to a 32-row bucket, so the
    // dynamic input shape changes once per bucket, capped at the capacity.
    std::int32_t active_rows(std::int32_t valid_rows) const;

  private:
    std::vector<DeviceTensor> keys_;
    std::vector<DeviceTensor> values_;
    std::int32_t num_layers_;
    std::int32_t num_kv_heads_;
    std::int32_t head_dim_;
    std::int32_t capacity_;
    DType dtype_;
    std::int32_t length_{0};
    std::int32_t write_index_{0};
    std::int32_t valid_rows_{0};
    std::vector<std::int32_t> positions_;
    std::vector<std::int32_t> key_positions_;
};

} // namespace trtmc::apertus
