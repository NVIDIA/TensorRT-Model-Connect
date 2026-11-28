/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Append-only KV cache of the Laguna engine.
//
// Cache row c of every layer holds the key / value of position c; the engine's
// sliding-window mask relies on that. The cache never wraps: a sequence that
// would exceed max_length is rejected before generation starts.

#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {
class ITrtModule;
}

namespace trtmc::laguna {

class KvCache {
  public:
    KvCache(std::int32_t num_layers, std::int32_t max_length, std::int32_t kv_width,
            cudaStream_t stream, DType dtype);

    bool ok() const;
    std::int32_t position() const { return position_; }
    std::int32_t max_length() const { return max_length_; }
    void reset() { position_ = 0; }

    // Binds cache inputs and single-row present outputs of the decode context.
    void bind_decode(ITrtModule& module);
    // Binds only the cache inputs of the prefill context; its present outputs
    // stay in the context's own buffers.
    void bind_prefill(ITrtModule& module);

    // token_id is supplied by the caller; this writes position_id and the
    // additive attention mask for `rows` new tokens at the current position.
    void prepare(TensorMap& inputs, std::int32_t rows);

    // Appends `rows` present rows of every layer at the current position.
    void append(const std::vector<const void*>& present_k,
                const std::vector<const void*>& present_v, std::int32_t rows);
    // Appends the bound single-row decode outputs.
    void append_decoded();

  private:
    std::int32_t num_layers_;
    std::int32_t max_length_;
    std::int32_t kv_width_;
    cudaStream_t stream_;
    std::size_t row_bytes_;
    std::int32_t position_{0};
    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    std::vector<DeviceTensor> present_k_;
    std::vector<DeviceTensor> present_v_;
    std::vector<float> mask_;
    std::vector<std::int32_t> positions_;
};

} // namespace trtmc::laguna
