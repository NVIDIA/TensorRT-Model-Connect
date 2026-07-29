/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Minimal lifecycle shared by GLM's runtime-owned native KV cache.

#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>

namespace trtmc {

class ITrtModule;
using TrtModule = ITrtModule;

class GlmInferenceState {
  public:
    virtual ~GlmInferenceState() = default;

    // --- Lifecycle ---

    // Reset logical state for a new sequence. Implementations may retain device
    // storage; native key-value lengths keep unwritten rows inactive.
    virtual void reset() = 0;

    // Bind all state tensors to the given TRT module.
    // Called once per sequence after reset(). The module reads/writes
    // state tensors via the bound device pointers.
    virtual void bind_to(TrtModule& module) = 0;

    // Write position and native KV scalar inputs before engine.forward().
    virtual void prepare_step(TensorMap& inputs, int32_t seq_len = 1) = 0;

    // Advance logical state after one decode step.
    virtual void advance(int32_t n_tokens = 1) = 0;

    // --- Queries ---

    // Current sequence position (0 = empty, increments with advance()).
    virtual int32_t position() const = 0;

    // Maximum sequence length this state can hold.
    // -1 for unbounded (recurrent models with no cache length limit).
    virtual int32_t max_length() const = 0;

    // Number of transformer/SSM layers.
    virtual int32_t num_layers() const = 0;

    // Total device memory consumed by this state (bytes).
    virtual std::size_t device_memory_bytes() const = 0;

    // Whether all allocations succeeded.
    virtual bool ok() const = 0;
};

} // namespace trtmc
