/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// BloomInferenceState: unified interface for autoregressive inference state.
//
// Both KV-cache attention state and recurrent state implementations expose
// this interface. Pipelines and plugins program against it — never against
// concrete state classes.
//
// The interface captures the lifecycle of per-sequence inference state:
//   1. reset()        — prepare for a new sequence
//   2. bind_to()      — bind state tensors to TRT engine I/O
//   3. prepare_step() — write state-related inputs (mask, position) into TensorMap
//   4. advance()      — update state after each decode step
//   5. position()     — current sequence position
//
// Implementations:
//   BloomKvCache          — dense append-only (current default)
//   Family-owned recurrent state — recurrent tensor state
//   Family-owned hybrid state — BloomKvCache + family-owned recurrent state composed
//   (future: RingKvCache, PagedKvCache, MlaCache, SlidingWindowCache)

#include "trtmc/runtime/tensor.h"

#include <cstddef>
#include <cstdint>

namespace trtmc {

class ITrtModule;
using TrtModule = ITrtModule;

class BloomInferenceState {
  public:
    virtual ~BloomInferenceState() = default;

    // --- Lifecycle ---

    // Reset state for a new sequence (zero buffers, position = 0).
    virtual void reset() = 0;

    // Bind all state tensors to the given TRT module.
    // Called once per sequence after reset(). The module reads/writes
    // state tensors via the bound device pointers.
    virtual void bind_to(TrtModule& module) = 0;

    // Write state-related inputs (mask, position, block table, etc.) into
    // the TensorMap before engine.forward(). The state owns its buffers —
    // Tensor.data pointers remain valid until the next prepare_step() call.
    // Pipelines call this instead of manually constructing mask/position tensors.
    virtual void prepare_step(TensorMap& inputs, int32_t seq_len = 1) = 0;

    // Update state after one decode step. Copies "present" outputs
    // into "cache" inputs, advances position.
    // n_tokens: number of tokens processed in this step (default 1).
    //           >1 for batched prefill / multi-token steps.
    virtual void advance(int32_t n_tokens = 1) = 0;

    // Provide the total prompt length before prefill starts.
    // Cache policies that distinguish prompt and decode tokens can use this
    // to protect prompt tokens even if compression triggers during prefill.
    virtual void set_prompt_length(int32_t prompt_length) { (void)prompt_length; }

    // Mark the transition from prompt prefill to autoregressive decoding.
    // State types that do not distinguish the phases can ignore this.
    virtual void mark_prefill_complete() {}

    // --- Queries ---

    // Current sequence position (0 = empty, increments with advance()).
    virtual int32_t position() const = 0;

    // Maximum sequence length this state can hold.
    // -1 for unbounded (recurrent models with no cache length limit).
    virtual int32_t max_length() const = 0;

    // Desired number of KV rows to expose to the decoder on the next step.
    // Dynamic-KV runtimes can use this to choose an execution profile/context
    // before prepare_step() binds the state tensors.
    virtual int32_t preferred_cache_rows() const { return max_length(); }

    // Number of transformer/SSM layers.
    virtual int32_t num_layers() const = 0;

    // Whether this state type needs an attention mask.
    // BloomKvCache -> true. Family-owned recurrent state -> false.
    virtual bool needs_attention_mask() const = 0;

    // Total device memory consumed by this state (bytes).
    virtual std::size_t device_memory_bytes() const = 0;

    // Human-readable state type for diagnostics.
    virtual const char* state_type() const = 0;

    // Whether all allocations succeeded.
    virtual bool ok() const = 0;
};

} // namespace trtmc
