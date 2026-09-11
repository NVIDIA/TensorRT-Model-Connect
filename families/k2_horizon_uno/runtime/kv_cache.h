/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

class ITrtModule;

struct K2HorizonUnoKvCacheNames {
    std::vector<std::string> cache_k;
    std::vector<std::string> cache_v;
    std::vector<std::string> present_k;
    std::vector<std::string> present_v;
    std::string cache_write_indices{"cache_write_indices"};
    std::string key_value_lengths{"key_value_lengths"};
    std::string position_id{"position_id"};
};

// Pure logical state used by the device-owning cache and focused CPU tests.
class K2HorizonUnoCacheCursor {
  public:
    explicit K2HorizonUnoCacheCursor(int32_t capacity);

    void validate_block(int32_t sequence_length) const;
    void advance(int32_t sequence_length);
    void rollback(int32_t position);
    void reset() { position_ = 0; }

    int32_t position() const { return position_; }
    int32_t capacity() const { return capacity_; }

  private:
    int32_t capacity_{0};
    int32_t position_{0};
};

// Fixed-capacity BF16 native KV cache. TensorRT updates each layer cache in
// place; present outputs must alias their cache inputs. Logical rollback hides
// temporary draft/verify suffixes, which later enqueues overwrite.
class K2HorizonUnoKvCache {
  public:
    K2HorizonUnoKvCache(int32_t max_length, cudaStream_t stream, K2HorizonUnoKvCacheNames names);

    void reset();
    void bind_to(ITrtModule& module);
    void prepare_block(TensorMap& inputs, int32_t sequence_length);
    void advance(int32_t sequence_length);
    void rollback(int32_t position);

    int32_t position() const { return cursor_.position(); }
    int32_t max_length() const { return cursor_.capacity(); }
    bool ok() const;

  private:
    void validate_engine_contract(ITrtModule& module) const;
    void bind_cache_aliases(ITrtModule& module);

    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    K2HorizonUnoCacheCursor cursor_;
    std::vector<int32_t> position_ids_;
    int32_t cache_write_index_{0};
    int32_t key_value_length_{0};
    K2HorizonUnoKvCacheNames names_;
    bool bound_{false};
};

} // namespace trtmc
