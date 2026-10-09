/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Qwen38MtpScheduler: drives MTP (multi-token-prediction) speculative
// decoding for one decode round -- draft num_draft_tokens() candidate
// tokens via the MTP draft-chain head in one call, verify all of them in
// one call to the (num_draft_tokens+1)-wide batched engine, and commit.
// Shared across Qwen3.8-27B / -FP8 / -NVFP4: precision is fully opaque
// here (baked into each engine's .plan at build time; this class only
// deals with tensor names/shapes, which do not change with precision or
// quantization).
//
// Commit has exactly two shapes, not a smooth gradient over how many
// drafts matched:
//
//   FULL ACCEPT (all num_draft_tokens() drafts confirmed): commit the
//   verify engine's own final state directly (append_prefill_kv() +
//   advance()) -- cheap, matches the old binary "accept" path exactly.
//
//   ANYTHING LESS (0 or more, but not all, drafts confirmed): the verify
//   engine's own conv/ssm (DeltaNet recurrent) state is NOT usable for
//   ANY partial prefix -- Qwen38RecurrentState only ever exposes the
//   FINAL post-all-substeps value, never an intermediate one, so there is
//   no way to extract "state as of the confirmed prefix" from it even
//   though the confirmed prefix's TOKENS and LOGITS are valid (causal:
//   each row's logits only depend on positions before it, so they're
//   trustworthy regardless of what happens at later, rejected rows).
//   The caller MUST instead re-run the confirmed-prefix length as
//   sequential single-token steps through the main decoder to rebuild
//   state safely -- this generalizes the old binary "reject" path (which
//   was just this rule's confirmed-length==1 special case).
//
// State ownership: this class owns MTP's own persistent attention state
// (a second, single-layer Qwen38KvCache) and reuses the main model's
// existing Qwen38HybridState for everything else. MTP's own cache is
// NEVER committed from draft_chain()'s speculative self-chained output --
// only from draft()'s single-step catch-up calls, seeded with REAL
// (verify-engine or re-run) hidden states, never MTP's own self-generated
// ones -- otherwise small approximation errors in self-chained hidden
// states would compound into MTP's persistent cache across rounds.

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
                       std::unique_ptr<ITrtModule> draft_chain_module,
                       std::unique_ptr<ITrtModule> multi_token_module, Qwen38HybridState& state,
                       int32_t hidden_size, int32_t vocab_size, int32_t num_draft_tokens,
                       cudaStream_t stream);

    // Bind MTP's own persistent cache to mtp_module_/draft_chain_module_,
    // and bind multi_token_module_'s per-layer state I/O directly to the
    // shared Qwen38HybridState's own buffers (cache_k/v inputs, conv/ssm
    // state inputs and outputs) plus this object's own present_k/v scratch
    // (shape (seq_len, kv_dim) -- too big for the main state's single-row
    // present buffers). Must be called once before draft()/draft_chain()/
    // verify_and_maybe_commit() are used, and is safe to call again after
    // a reset() (device pointers are stable across
    // Qwen38KvCache::reset()/Qwen38RecurrentState::reset()).
    void bind_state();

    // Reset MTP's own KV cache for a new sequence. Does not touch the
    // shared main-model state -- the caller resets that separately.
    void reset();

    // Run one MTP forward step for `token` at `position`, using a REAL
    // (never MTP's own self-chained) hidden state -- the main engine's
    // hidden_state output from the call that just produced `token`, or
    // the verify engine's per-row hidden_states output for a confirmed
    // position. Used for prefill warm-up (known prompt tokens) and for
    // resyncing MTP's persistent cache after verification confirms a
    // newly-accepted token (catch-up calls). Returns the argmax draft
    // token id. Advances MTP's own cache by one step -- must be called
    // exactly once per position in the main model's token stream, in
    // order, or MTP's attention history desyncs.
    int32_t draft(int32_t token, int32_t position, const float* hidden_state);

    struct DraftChainResult {
        std::vector<int32_t> token_ids;   // (num_draft_tokens(),)
        std::vector<float> hidden_states; // flattened (num_draft_tokens(), hidden_size())
    };

    // Draft num_draft_tokens() candidate tokens in ONE call, by repeating
    // the MTP layer num_draft_tokens() times in-graph (in-graph argmax +
    // self-chained hidden state between repeats -- see
    // build_mtp_draft_chain_engine). Does NOT touch MTP's own persistent
    // cache (mtp_cache_) -- its own present_k/v outputs are discarded;
    // resyncing mtp_cache_ for whatever prefix verification confirms is
    // done separately via draft() catch-up calls using REAL hidden
    // states, never this call's self-chained approximations.
    DraftChainResult draft_chain(int32_t token, int32_t position, const float* hidden_state);

    struct VerifyResult {
        // Number of candidates confirmed, counting real_next: always >=1
        // (real_next is never in question -- only drafts are verified)
        // and <= num_draft_tokens()+1. == num_draft_tokens()+1 means
        // FULL ACCEPT; anything less is PARTIAL -- see class-level
        // comment for why these need structurally different recovery.
        int32_t accepted_length{1};
        bool full_accept{false};
        // Confirmed token ids [real_next, draft_0, ...], length ==
        // accepted_length. Always populated regardless of full_accept --
        // read directly from argmax of verify's own per-row logits, valid
        // even when the corresponding STATE isn't (causal: row i's logits
        // only depend on positions before i).
        std::vector<int32_t> accepted_tokens;
        // Row (accepted_length-1)'s argmax: the model's own "free"
        // prediction for what comes right after the confirmed prefix.
        // Always populated and always trustworthy as a value.
        int32_t next_real_candidate{0};
        // Only populated when full_accept: verify engine's own per-row
        // hidden_states for ALL accepted_length rows (== seq_len in this
        // case) -- rows [0, accepted_length-1) seed the catch-up draft()
        // calls for each newly-confirmed draft, row accepted_length-1
        // seeds the next round's draft_chain() call. When NOT full_accept
        // this is empty -- the caller gets real hidden states instead
        // from accepted_length sequential run_step() calls.
        std::vector<float> hidden_states; // flattened (accepted_length, hidden_size())
    };

    // Verify [real_next, draft_tokens...] (draft_tokens.size() must equal
    // num_draft_tokens()) at positions [base_step+1, base_step+seq_len())
    // against the seq_len()-wide batched verification engine (base_step is
    // the last committed index -- the shared state's kv position must
    // equal base_step+1 on entry). On full accept, commits all seq_len()
    // tokens into the shared Qwen38HybridState via
    // append_prefill_kv()/advance(). On anything less, the shared state is
    // left completely untouched -- caller must recover via accepted_length
    // sequential run_step() calls (see class-level comment).
    VerifyResult verify_and_maybe_commit(int32_t real_next,
                                         const std::vector<int32_t>& draft_tokens,
                                         int32_t base_step);

    int32_t hidden_size() const { return hidden_size_; }
    int32_t vocab_size() const { return vocab_size_; }
    int32_t num_draft_tokens() const { return num_draft_tokens_; }
    int32_t seq_len() const { return num_draft_tokens_ + 1; }

  private:
    std::unique_ptr<ITrtModule> mtp_module_;
    std::unique_ptr<ITrtModule> draft_chain_module_;
    std::unique_ptr<ITrtModule> multi_token_module_;
    Qwen38HybridState& state_;
    cudaStream_t stream_;
    int32_t hidden_size_;
    int32_t vocab_size_;
    int32_t num_draft_tokens_;

    std::unique_ptr<Qwen38KvCache> mtp_cache_;

    // Scratch for the multi-token engine's (seq_len, kv_dim) present_k/v
    // outputs per attention layer -- committed into the shared cache via
    // append_prefill_kv() on full accept.
    std::vector<DeviceTensor> multi_present_k_;
    std::vector<DeviceTensor> multi_present_v_;

    // Scratch for draft_chain_module_'s present_k/v outputs. Allocated
    // because TensorRT requires every marked output to be bound before
    // execute, even though this class never reads these back -- see the
    // class-level comment on why draft_chain()'s own cache state is
    // discarded rather than committed.
    std::vector<DeviceTensor> draft_chain_present_k_;
    std::vector<DeviceTensor> draft_chain_present_v_;

    void* mtp_logits_device_ptr_{nullptr};
    void* draft_chain_ids_device_ptr_{nullptr};
    void* draft_chain_hidden_device_ptr_{nullptr};
    void* draft_chain_logits_device_ptr_{nullptr};
    void* multi_logits_device_ptr_{nullptr};
    void* multi_hidden_device_ptr_{nullptr};
    std::vector<float> mtp_logits_host_;
    std::vector<float> multi_logits_host_;
    std::vector<float> mask_scratch_;
    bool bound_{false};
};

} // namespace trtmc
