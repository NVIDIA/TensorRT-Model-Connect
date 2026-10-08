/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_8/runtime/mtp_scheduler.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trtmc {

namespace {
constexpr float kMaskedScore = -1.0e4F;

int32_t argmax(const float* data, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (data[i] > data[best])
            best = i;
    }
    return static_cast<int32_t>(best);
}
} // namespace

Qwen38MtpScheduler::Qwen38MtpScheduler(std::unique_ptr<ITrtModule> mtp_module,
                                       std::unique_ptr<ITrtModule> draft_chain_module,
                                       std::unique_ptr<ITrtModule> multi_token_module,
                                       Qwen38HybridState& state, int32_t hidden_size,
                                       int32_t vocab_size, int32_t num_draft_tokens,
                                       cudaStream_t stream)
    : mtp_module_(std::move(mtp_module)), draft_chain_module_(std::move(draft_chain_module)),
      multi_token_module_(std::move(multi_token_module)), state_(state), stream_(stream),
      hidden_size_(hidden_size), vocab_size_(vocab_size), num_draft_tokens_(num_draft_tokens) {
    if (!mtp_module_ || !mtp_module_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: invalid MTP module");
    if (!draft_chain_module_ || !draft_chain_module_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: invalid MTP draft-chain module");
    if (!multi_token_module_ || !multi_token_module_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: invalid multi-token module");
    if (num_draft_tokens_ < 1)
        throw std::runtime_error("Qwen38MtpScheduler: num_draft_tokens must be >= 1");

    Qwen38KvCache& main_kv = *state_.kv_cache();

    Qwen38KvCacheNames mtp_names;
    mtp_names.cache_k = {"mtp_cache_k"};
    mtp_names.cache_v = {"mtp_cache_v"};
    mtp_names.present_k = {"mtp_present_k"};
    mtp_names.present_v = {"mtp_present_v"};
    mtp_cache_ = std::make_unique<Qwen38KvCache>(1, main_kv.max_length(), main_kv.kv_dim(), stream_,
                                                 main_kv.dtype(), mtp_names);
    if (!mtp_cache_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: failed to allocate MTP's own KV cache");

    const int32_t seq_len_val = num_draft_tokens_ + 1;
    multi_present_k_.reserve(static_cast<std::size_t>(main_kv.num_layers()));
    multi_present_v_.reserve(static_cast<std::size_t>(main_kv.num_layers()));
    for (int32_t i = 0; i < main_kv.num_layers(); ++i) {
        multi_present_k_.emplace_back(std::vector<int64_t>{seq_len_val, main_kv.kv_dim()},
                                      main_kv.dtype(), stream_);
        multi_present_v_.emplace_back(std::vector<int64_t>{seq_len_val, main_kv.kv_dim()},
                                      main_kv.dtype(), stream_);
        if (!multi_present_k_.back().ok() || !multi_present_v_.back().ok())
            throw std::runtime_error(
                "Qwen38MtpScheduler: failed to allocate multi-token present K/V scratch");
    }

    // Single entry each -- draft_chain_module_'s present_k/v are not
    // per-main-model-layer, they're MTP's own one layer (matching
    // build_mtp_draft_chain_engine's singular mtp_present_k/v names).
    draft_chain_present_k_.emplace_back(
        std::vector<int64_t>{num_draft_tokens_, main_kv.kv_dim()}, main_kv.dtype(), stream_);
    draft_chain_present_v_.emplace_back(
        std::vector<int64_t>{num_draft_tokens_, main_kv.kv_dim()}, main_kv.dtype(), stream_);
    if (!draft_chain_present_k_.back().ok() || !draft_chain_present_v_.back().ok()) {
        throw std::runtime_error(
            "Qwen38MtpScheduler: failed to allocate draft-chain present K/V scratch");
    }
}

void Qwen38MtpScheduler::bind_state() {
    mtp_cache_->bind_to(*mtp_module_);

    // draft_chain_module_ reads MTP's persistent cache (read-only, same
    // convention as the multi-token engine reading the main model's
    // cache) but its own present_k/v outputs are never committed back --
    // bind them to throwaway scratch (TensorRT requires every marked
    // output bound before execute, even unused ones).
    mtp_cache_->bind_cache_inputs(*draft_chain_module_);
    draft_chain_module_->bind_external("mtp_present_k", draft_chain_present_k_[0].data());
    draft_chain_module_->bind_external("mtp_present_v", draft_chain_present_v_[0].data());

    Qwen38KvCache& main_kv = *state_.kv_cache();
    Qwen38RecurrentState& ssm = *state_.recurrent_state();

    main_kv.bind_cache_inputs(*multi_token_module_);

    for (int32_t i = 0; i < main_kv.num_layers(); ++i) {
        const auto suffix = "_" + std::to_string(i);
        multi_token_module_->bind_external("present_k" + suffix,
                                           multi_present_k_[static_cast<std::size_t>(i)].data());
        multi_token_module_->bind_external("present_v" + suffix,
                                           multi_present_v_[static_cast<std::size_t>(i)].data());
    }

    for (int32_t i = 0; i < ssm.num_layers(); ++i) {
        const auto suffix = "_" + std::to_string(i);
        multi_token_module_->bind_external("conv_state" + suffix, ssm.state_ptr(0, i));
        multi_token_module_->bind_external("ssm_state" + suffix, ssm.state_ptr(1, i));
        multi_token_module_->bind_external("present_conv" + suffix, ssm.present_ptr(0, i));
        multi_token_module_->bind_external("present_ssm" + suffix, ssm.present_ptr(1, i));
    }

    bound_ = true;
}

void Qwen38MtpScheduler::reset() {
    mtp_cache_->reset();
}

int32_t Qwen38MtpScheduler::draft(int32_t token, int32_t position, const float* hidden_state) {
    if (!bound_)
        throw std::runtime_error("Qwen38MtpScheduler: bind_state() must be called before draft()");
    // mtp_cache_->position() tracks CALL COUNT (how many rows have been
    // written to MTP's own cache so far -- used for mask validity and
    // cache row indexing), not the RoPE position of the token being
    // embedded. MTP's first-ever call always embeds the token at absolute
    // sequence position 1 (the main model embeds position 0 by itself,
    // with no MTP involvement), so the invariant is call_count ==
    // position - 1, not call_count == position.
    if (mtp_cache_->position() != position - 1) {
        throw std::runtime_error(
            "Qwen38MtpScheduler::draft: MTP cache position out of sync with caller's position "
            "(every main-model token must get exactly one draft()/warm-up call, in order)");
    }

    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["next_token_id"] = token_t;

    Tensor hidden_t;
    hidden_t.data = const_cast<float*>(hidden_state);
    hidden_t.shape = {1, hidden_size_};
    hidden_t.dtype = DType::kFloat32;
    inputs["hidden_state"] = hidden_t;

    // prepare_step() writes both attention_mask (correct: call-count-based
    // validity) and position_id (WRONG here: it writes mtp_cache_'s own
    // call-count, not the caller's absolute `position`) -- overwrite
    // position_id right after with the correct value.
    mtp_cache_->prepare_step(inputs, 1);
    Tensor position_t;
    position_t.data = &position;
    position_t.shape = {1};
    position_t.dtype = DType::kInt32;
    inputs["position_id"] = position_t;

    mtp_module_->forward_async(inputs);
    mtp_module_->sync();

    if (mtp_logits_device_ptr_ == nullptr) {
        mtp_logits_device_ptr_ = mtp_module_->device_ptr("mtp_logits");
        if (mtp_logits_device_ptr_ == nullptr)
            throw std::runtime_error("Qwen38MtpScheduler: MTP module has no 'mtp_logits' output");
    }
    mtp_logits_host_.resize(static_cast<std::size_t>(vocab_size_));
    const cudaError_t copy_status =
        cudaMemcpy(mtp_logits_host_.data(), mtp_logits_device_ptr_,
                  mtp_logits_host_.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen38MtpScheduler: failed to copy mtp_logits: ") +
                                 cudaGetErrorString(copy_status));

    mtp_cache_->advance(1);

    return argmax(mtp_logits_host_.data(), mtp_logits_host_.size());
}

Qwen38MtpScheduler::DraftChainResult
Qwen38MtpScheduler::draft_chain(int32_t token, int32_t position, const float* hidden_state) {
    if (!bound_) {
        throw std::runtime_error(
            "Qwen38MtpScheduler: bind_state() must be called before draft_chain()");
    }

    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["next_token_id"] = token_t;

    Tensor pos_t;
    pos_t.data = &position;
    pos_t.shape = {1};
    pos_t.dtype = DType::kInt32;
    inputs["position_id"] = pos_t;

    Tensor hidden_t;
    hidden_t.data = const_cast<float*>(hidden_state);
    hidden_t.shape = {1, hidden_size_};
    hidden_t.dtype = DType::kFloat32;
    inputs["hidden_state"] = hidden_t;

    // Same persistent-prefix mask convention as draft()'s
    // mtp_cache_->prepare_step() builds (shape (1, max_length+1), valid
    // prefix + the final "self" column always valid) -- built directly
    // here since this module takes position_id as a plain input, with no
    // auto-write to override afterward like draft() needs.
    const int32_t max_len = mtp_cache_->max_length();
    mask_scratch_.assign(static_cast<std::size_t>(max_len) + 1, kMaskedScore);
    const int32_t valid = std::min(mtp_cache_->position(), max_len);
    std::fill(mask_scratch_.begin(), mask_scratch_.begin() + valid, 0.0F);
    mask_scratch_[static_cast<std::size_t>(max_len)] = 0.0F;
    Tensor mask_t;
    mask_t.data = mask_scratch_.data();
    mask_t.shape = {1, max_len + 1};
    mask_t.dtype = DType::kFloat32;
    inputs["attention_mask"] = mask_t;

    draft_chain_module_->forward_async(inputs);
    draft_chain_module_->sync();

    if (draft_chain_ids_device_ptr_ == nullptr) {
        draft_chain_ids_device_ptr_ = draft_chain_module_->device_ptr("mtp_draft_token_ids");
        if (draft_chain_ids_device_ptr_ == nullptr) {
            throw std::runtime_error(
                "Qwen38MtpScheduler: draft-chain module has no 'mtp_draft_token_ids' output");
        }
    }
    if (draft_chain_hidden_device_ptr_ == nullptr) {
        draft_chain_hidden_device_ptr_ = draft_chain_module_->device_ptr("mtp_draft_hidden_states");
        if (draft_chain_hidden_device_ptr_ == nullptr) {
            throw std::runtime_error(
                "Qwen38MtpScheduler: draft-chain module has no 'mtp_draft_hidden_states' output");
        }
    }

    DraftChainResult result;
    result.token_ids.resize(static_cast<std::size_t>(num_draft_tokens_));
    cudaError_t copy_status =
        cudaMemcpy(result.token_ids.data(), draft_chain_ids_device_ptr_,
                  result.token_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(std::string("Qwen38MtpScheduler: failed to copy draft token ids: ") +
                                 cudaGetErrorString(copy_status));
    }

    result.hidden_states.resize(static_cast<std::size_t>(num_draft_tokens_) *
                                static_cast<std::size_t>(hidden_size_));
    copy_status = cudaMemcpy(result.hidden_states.data(), draft_chain_hidden_device_ptr_,
                            result.hidden_states.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(
            std::string("Qwen38MtpScheduler: failed to copy draft hidden states: ") +
            cudaGetErrorString(copy_status));
    }

    return result;
}

Qwen38MtpScheduler::VerifyResult
Qwen38MtpScheduler::verify_and_maybe_commit(int32_t real_next,
                                            const std::vector<int32_t>& draft_tokens,
                                            int32_t base_step) {
    if (!bound_) {
        throw std::runtime_error(
            "Qwen38MtpScheduler: bind_state() must be called before verify_and_maybe_commit()");
    }
    if (static_cast<int32_t>(draft_tokens.size()) != num_draft_tokens_) {
        throw std::runtime_error(
            "Qwen38MtpScheduler::verify_and_maybe_commit: draft_tokens.size() must equal "
            "num_draft_tokens()");
    }

    Qwen38KvCache& kv = *state_.kv_cache();
    Qwen38RecurrentState& ssm = *state_.recurrent_state();

    if (kv.position() != base_step + 1) {
        throw std::runtime_error(
            "Qwen38MtpScheduler::verify_and_maybe_commit: shared state position out of sync "
            "with base_step");
    }

    const int32_t n = seq_len();
    std::vector<int32_t> tokens(static_cast<std::size_t>(n));
    std::vector<int32_t> positions(static_cast<std::size_t>(n));
    tokens[0] = real_next;
    positions[0] = base_step + 1;
    for (int32_t i = 0; i < num_draft_tokens_; ++i) {
        tokens[static_cast<std::size_t>(i) + 1] = draft_tokens[static_cast<std::size_t>(i)];
        positions[static_cast<std::size_t>(i) + 1] = base_step + 2 + i;
    }

    TensorMap inputs;

    Tensor tok_t;
    tok_t.data = tokens.data();
    tok_t.shape = {n};
    tok_t.dtype = DType::kInt32;
    inputs["token_ids"] = tok_t;

    Tensor pos_t;
    pos_t.data = positions.data();
    pos_t.shape = {n};
    pos_t.dtype = DType::kInt32;
    inputs["position_ids"] = pos_t;

    // mask_persistent_only(valid = kv.position()): only the already-
    // committed prefix is visible; the n new tokens' causal visibility is
    // handled internally by the engine's own per-substep mask extension.
    const int32_t max_len = kv.max_length();
    mask_scratch_.assign(static_cast<std::size_t>(max_len), kMaskedScore);
    const int32_t valid = std::min(kv.position(), max_len);
    std::fill(mask_scratch_.begin(), mask_scratch_.begin() + valid, 0.0F);
    Tensor mask_t;
    mask_t.data = mask_scratch_.data();
    mask_t.shape = {1, max_len};
    mask_t.dtype = DType::kFloat32;
    inputs["attention_mask"] = mask_t;

    multi_token_module_->forward_async(inputs);
    multi_token_module_->sync();

    if (multi_logits_device_ptr_ == nullptr) {
        multi_logits_device_ptr_ = multi_token_module_->device_ptr("logits");
        if (multi_logits_device_ptr_ == nullptr)
            throw std::runtime_error("Qwen38MtpScheduler: multi-token module has no 'logits' output");
    }
    multi_logits_host_.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(vocab_size_));
    cudaError_t copy_status =
        cudaMemcpy(multi_logits_host_.data(), multi_logits_device_ptr_,
                  multi_logits_host_.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen38MtpScheduler: failed to copy logits: ") +
                                 cudaGetErrorString(copy_status));

    // --- longest-prefix-match ---
    VerifyResult result;
    result.accepted_tokens.push_back(real_next);
    int32_t accepted_drafts = 0;
    for (int32_t i = 0; i < num_draft_tokens_; ++i) {
        const int32_t row_argmax = argmax(
            multi_logits_host_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(vocab_size_),
            static_cast<std::size_t>(vocab_size_));
        if (row_argmax != draft_tokens[static_cast<std::size_t>(i)]) {
            result.next_real_candidate = row_argmax;
            break;
        }
        result.accepted_tokens.push_back(draft_tokens[static_cast<std::size_t>(i)]);
        ++accepted_drafts;
    }
    result.accepted_length = accepted_drafts + 1;
    result.full_accept = (accepted_drafts == num_draft_tokens_);

    if (result.full_accept) {
        // No mismatch broke the loop above -- the "free" next-round
        // candidate is row (n-1)'s argmax, not yet computed.
        result.next_real_candidate = argmax(
            multi_logits_host_.data() +
                static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(vocab_size_),
            static_cast<std::size_t>(vocab_size_));
    }

    if (!result.full_accept)
        return result; // shared state untouched; caller recovers via run_step()

    // --- full accept: commit directly. append_prefill_kv()/advance() are
    // already-correct, existing multi-token-aware methods -- no new
    // cache-indexing math here. ---
    std::vector<const void*> pk;
    std::vector<const void*> pv;
    pk.reserve(static_cast<std::size_t>(kv.num_layers()));
    pv.reserve(static_cast<std::size_t>(kv.num_layers()));
    for (int32_t i = 0; i < kv.num_layers(); ++i) {
        pk.push_back(multi_present_k_[static_cast<std::size_t>(i)].data());
        pv.push_back(multi_present_v_[static_cast<std::size_t>(i)].data());
    }
    kv.append_prefill_kv(pk, pv, n);
    ssm.advance(n);

    if (multi_hidden_device_ptr_ == nullptr) {
        multi_hidden_device_ptr_ = multi_token_module_->device_ptr("hidden_states");
        if (multi_hidden_device_ptr_ == nullptr) {
            throw std::runtime_error(
                "Qwen38MtpScheduler: multi-token module has no 'hidden_states' output");
        }
    }
    result.hidden_states.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(hidden_size_));
    copy_status = cudaMemcpy(result.hidden_states.data(), multi_hidden_device_ptr_,
                            result.hidden_states.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess) {
        throw std::runtime_error(std::string("Qwen38MtpScheduler: failed to copy hidden_states: ") +
                                 cudaGetErrorString(copy_status));
    }

    return result;
}

} // namespace trtmc
