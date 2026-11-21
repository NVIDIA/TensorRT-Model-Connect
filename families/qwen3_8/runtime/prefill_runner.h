/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Qwen38PrefillRunner: drives the multi-token prefill engine. One call
// consumes up to chunk_len() prompt tokens with batched GEMMs, instead of one
// single-token decoder step per token.
//
// It shares the main model's Qwen38HybridState (KV cache + DeltaNet conv/ssm
// state) with the decoder: the prefill engine reads the same cache_k/v and
// conv/ssm state buffers and writes its conv/ssm results into the state's own
// "present" buffers, so a plain recurrent-state advance(n) commits them. Its
// K/V outputs have chunk_len() rows per layer and go to private scratch; only
// the first n (valid) rows are appended to the cache.
//
// The last chunk of a prompt may be partial (n < chunk_len()). The engine
// takes valid_len=n and keeps padding rows from touching the recurrent state,
// so no special handling is needed here beyond passing n.

#include "families/qwen3_8/runtime/hybrid_state.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

class Qwen38PrefillRunner {
  public:
    Qwen38PrefillRunner(std::unique_ptr<ITrtModule> module, Qwen38HybridState& state,
                        int32_t chunk_len, int32_t hidden_size, int32_t vocab_size,
                        cudaStream_t stream);

    // Bind the shared state's buffers and this runner's scratch to the engine.
    // Call once per generation, after the state is reset. Does not touch the
    // KV cache's position-input bookkeeping (unlike Qwen38KvCache::bind_to).
    void bind_state();

    int32_t chunk_len() const { return chunk_len_; }

    // True if a chunk of n tokens still fits in the KV cache.
    bool fits(int32_t n) const;

    // Process tokens[0, n) (1 <= n <= chunk_len()) as the next n positions of
    // the sequence, then commit KV and recurrent state. `logits` receives the
    // vocab-sized logits of the LAST token. If hidden_rows is non-null it
    // receives n * hidden_size final-normed hidden states, one row per token.
    void run_chunk(const int32_t* tokens, int32_t n, std::vector<float>& logits,
                   std::vector<float>* hidden_rows = nullptr);

  private:
    std::unique_ptr<ITrtModule> module_;
    Qwen38HybridState& state_;
    cudaStream_t stream_;
    int32_t chunk_len_;
    int32_t hidden_size_;
    int32_t vocab_size_;

    std::vector<DeviceTensor> present_k_;
    std::vector<DeviceTensor> present_v_;

    std::vector<int32_t> tokens_buf_;
    std::vector<int32_t> positions_buf_;
    std::vector<float> mask_buf_;
    void* logits_ptr_{nullptr};
    void* hidden_ptr_{nullptr};
    bool bound_{false};
};

} // namespace trtmc
