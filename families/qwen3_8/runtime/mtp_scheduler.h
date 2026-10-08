/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Qwen38MtpScheduler: drives MTP (multi-token-prediction) speculative
// decoding for one decode round -- draft a token via the MTP head, verify
// it with a 2-token batched engine, and commit on accept. Shared across
// Qwen3.8-27B / -FP8 / -NVFP4: precision is fully opaque here (baked into
// each engine's .plan at build time; this class only deals with tensor
// names/shapes, which do not change with precision or quantization).
//
// State ownership: this class owns MTP's own persistent attention state
// (a second, single-layer Qwen38KvCache -- MTP's one decoder layer has a
// real, continuously-updated KV cache, not a stateless lookup) and reuses
// the main model's existing Qwen38HybridState for everything else. On
// accept, this class commits into that shared state via
// Qwen38KvCache::append_prefill_kv() and Qwen38RecurrentState::advance(),
// which are already correct for a multi-token write (no new cache-indexing
// math here, deliberately, to avoid re-introducing an off-by-one).
//
// Reject is NOT handled inside this class: on reject the caller must run
// an ordinary single-token step through the main decoder engine, exactly
// like the non-speculative decode path -- this class does not own the main
// decoder module.

#include "families/qwen3_8/runtime/hybrid_state.h"
#include "families/qwen3_8/runtime/kv_cache.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

class Qwen38MtpScheduler {
  public:
    Qwen38MtpScheduler(std::unique_ptr<ITrtModule> mtp_module,
                       std::unique_ptr<ITrtModule> multi_token_module, Qwen38HybridState& state,
                       int32_t hidden_size, int32_t vocab_size, cudaStream_t stream);

    // Bind MTP's own persistent cache to mtp_module_, and bind
    // multi_token_module_'s per-layer state I/O directly to the shared
    // Qwen38HybridState's own buffers (cache_k/v inputs, conv/ssm state
    // inputs and outputs) plus this object's own present_k/v scratch
    // (shape (2, kv_dim) -- too big for the main state's single-row
    // present buffers). Must be called once before draft()/verify() are
    // used, and is safe to call again after a reset() (device pointers are
    // stable across Qwen38KvCache::reset()/Qwen38RecurrentState::reset()).
    void bind_state();

    // Reset MTP's own KV cache for a new sequence. Does not touch the
    // shared main-model state -- the caller resets that separately.
    void reset();

    // Run one MTP forward step for `token` at `position`, using the main
    // engine's hidden_state output from the call that just produced
    // `token`. Used both for prefill warm-up (known prompt tokens) and for
    // drafting (the just-confirmed real token). Returns the argmax draft
    // token id. Advances MTP's own cache by one step -- must be called
    // exactly once per position in the main model's token stream, in
    // order, or MTP's attention history desyncs.
    int32_t draft(int32_t token, int32_t position, const float* hidden_state);

    struct VerifyResult {
        bool accepted{false};
        int32_t verified_token{0};
        // Only populated when accepted: the multi-token engine's own
        // per-row hidden_states output, needed for the next round's two
        // draft() calls (mirrors the Python driver's h1/h2).
        std::vector<float> hidden_row0;
        std::vector<float> hidden_row1;
        // Only populated when accepted: argmax of row 1's logits -- the
        // prediction for the token AFTER `draft`, which this same verify
        // call already computed for free (no extra engine call needed to
        // get the next round's real_next candidate).
        int32_t next_real_candidate{0};
    };

    // Verify [real_next, draft] at positions [base_step+1, base_step+2]
    // against the 2-token batched engine (base_step is the last committed
    // index -- the shared state's kv position must equal base_step+1 on
    // entry). On accept, commits both tokens into the shared
    // Qwen38HybridState via append_prefill_kv()/advance(2) and returns the
    // per-row hidden states. On reject, the shared state is left
    // completely untouched.
    VerifyResult verify_and_maybe_commit(int32_t real_next, int32_t draft, int32_t base_step);

    int32_t hidden_size() const { return hidden_size_; }
    int32_t vocab_size() const { return vocab_size_; }

  private:
    std::unique_ptr<ITrtModule> mtp_module_;
    std::unique_ptr<ITrtModule> multi_token_module_;
    Qwen38HybridState& state_;
    cudaStream_t stream_;
    int32_t hidden_size_;
    int32_t vocab_size_;

    std::unique_ptr<Qwen38KvCache> mtp_cache_;

    // Scratch for the multi-token engine's (2, kv_dim) present_k/v outputs
    // per attention layer -- committed into the shared cache via
    // append_prefill_kv() on accept.
    std::vector<DeviceTensor> multi_present_k_;
    std::vector<DeviceTensor> multi_present_v_;

    void* mtp_logits_device_ptr_{nullptr};
    void* multi_logits_device_ptr_{nullptr};
    void* multi_hidden_device_ptr_{nullptr};
    std::vector<float> mtp_logits_host_;
    std::vector<float> multi_logits_host_;
    std::vector<float> mask_scratch_;
    bool bound_{false};
};

} // namespace trtmc
