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
                                       std::unique_ptr<ITrtModule> multi_token_module,
                                       Qwen38HybridState& state, int32_t hidden_size,
                                       int32_t vocab_size, cudaStream_t stream)
    : mtp_module_(std::move(mtp_module)), multi_token_module_(std::move(multi_token_module)),
      state_(state), stream_(stream), hidden_size_(hidden_size), vocab_size_(vocab_size) {
    if (!mtp_module_ || !mtp_module_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: invalid MTP module");
    if (!multi_token_module_ || !multi_token_module_->ok())
        throw std::runtime_error("Qwen38MtpScheduler: invalid multi-token module");

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

    multi_present_k_.reserve(static_cast<std::size_t>(main_kv.num_layers()));
    multi_present_v_.reserve(static_cast<std::size_t>(main_kv.num_layers()));
    for (int32_t i = 0; i < main_kv.num_layers(); ++i) {
        multi_present_k_.emplace_back(std::vector<int64_t>{2, main_kv.kv_dim()}, main_kv.dtype(),
                                      stream_);
        multi_present_v_.emplace_back(std::vector<int64_t>{2, main_kv.kv_dim()}, main_kv.dtype(),
                                      stream_);
        if (!multi_present_k_.back().ok() || !multi_present_v_.back().ok())
            throw std::runtime_error(
                "Qwen38MtpScheduler: failed to allocate multi-token present K/V scratch");
    }
}

void Qwen38MtpScheduler::bind_state() {
    mtp_cache_->bind_to(*mtp_module_);

    Qwen38KvCache& main_kv = *state_.kv_cache();
    Qwen38RecurrentState& ssm = *state_.recurrent_state();

    // cache_k/v INPUTS: shared, read-only view of the main model's own
    // cache (the multi-token engine's present_k/v outputs are (2, kv_dim),
    // too big for Qwen38KvCache::bind_to()'s single-row present buffers,
    // so that method is deliberately not used here).
    main_kv.bind_cache_inputs(*multi_token_module_);

    for (int32_t i = 0; i < main_kv.num_layers(); ++i) {
        const auto suffix = "_" + std::to_string(i);
        multi_token_module_->bind_external("present_k" + suffix,
                                           multi_present_k_[static_cast<std::size_t>(i)].data());
        multi_token_module_->bind_external("present_v" + suffix,
                                           multi_present_v_[static_cast<std::size_t>(i)].data());
    }

    // conv_state/ssm_state INPUTS and present_conv/present_ssm OUTPUTS bind
    // directly to the shared recurrent state's own buffers -- running this
    // engine overwrites the same present_ buffers the main single-token
    // engine writes, so a plain Qwen38RecurrentState::advance(2) afterward
    // picks up this engine's results with no extra copying.
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
    // written to MTP's own cache so far -- used for mask validity and cache
    // row indexing), not the RoPE position of the token being embedded.
    // MTP's first-ever call always embeds the token at absolute sequence
    // position 1 (the main model embeds position 0 by itself, with no MTP
    // involvement), so the invariant is call_count == position - 1, not
    // call_count == position.
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

Qwen38MtpScheduler::VerifyResult
Qwen38MtpScheduler::verify_and_maybe_commit(int32_t real_next, int32_t draft, int32_t base_step) {
    if (!bound_) {
        throw std::runtime_error(
            "Qwen38MtpScheduler: bind_state() must be called before verify_and_maybe_commit()");
    }
    Qwen38KvCache& kv = *state_.kv_cache();
    Qwen38RecurrentState& ssm = *state_.recurrent_state();

    if (kv.position() != base_step + 1) {
        throw std::runtime_error(
            "Qwen38MtpScheduler::verify_and_maybe_commit: shared state position out of sync "
            "with base_step");
    }

    const int32_t p0 = base_step + 1;
    const int32_t p1 = base_step + 2;
    int32_t tokens[2] = {real_next, draft};
    int32_t positions[2] = {p0, p1};

    TensorMap inputs;

    Tensor tok_t;
    tok_t.data = tokens;
    tok_t.shape = {2};
    tok_t.dtype = DType::kInt32;
    inputs["token_ids"] = tok_t;

    Tensor pos_t;
    pos_t.data = positions;
    pos_t.shape = {2};
    pos_t.dtype = DType::kInt32;
    inputs["position_ids"] = pos_t;

    // mask_persistent_only(valid = kv.position()): only the already-
    // committed prefix is visible; the 2 new tokens' causal visibility is
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
    multi_logits_host_.resize(2 * static_cast<std::size_t>(vocab_size_));
    cudaError_t copy_status =
        cudaMemcpy(multi_logits_host_.data(), multi_logits_device_ptr_,
                  multi_logits_host_.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen38MtpScheduler: failed to copy logits: ") +
                                 cudaGetErrorString(copy_status));

    VerifyResult result;
    result.verified_token =
        argmax(multi_logits_host_.data(), static_cast<std::size_t>(vocab_size_));
    result.accepted = (result.verified_token == draft);

    if (!result.accepted)
        return result;

    // Commit: append_prefill_kv()/advance(2) are already-correct, existing
    // multi-token-aware methods -- no new cache-indexing math here.
    std::vector<const void*> pk;
    std::vector<const void*> pv;
    pk.reserve(static_cast<std::size_t>(kv.num_layers()));
    pv.reserve(static_cast<std::size_t>(kv.num_layers()));
    for (int32_t i = 0; i < kv.num_layers(); ++i) {
        pk.push_back(multi_present_k_[static_cast<std::size_t>(i)].data());
        pv.push_back(multi_present_v_[static_cast<std::size_t>(i)].data());
    }
    kv.append_prefill_kv(pk, pv, 2);
    ssm.advance(2);

    if (multi_hidden_device_ptr_ == nullptr) {
        multi_hidden_device_ptr_ = multi_token_module_->device_ptr("hidden_states");
        if (multi_hidden_device_ptr_ == nullptr) {
            throw std::runtime_error(
                "Qwen38MtpScheduler: multi-token module has no 'hidden_states' output");
        }
    }
    result.hidden_row0.resize(static_cast<std::size_t>(hidden_size_));
    result.hidden_row1.resize(static_cast<std::size_t>(hidden_size_));
    const auto row_bytes = static_cast<std::size_t>(hidden_size_) * sizeof(float);
    cudaMemcpy(result.hidden_row0.data(), multi_hidden_device_ptr_, row_bytes,
              cudaMemcpyDeviceToHost);
    cudaMemcpy(result.hidden_row1.data(),
              static_cast<const uint8_t*>(multi_hidden_device_ptr_) + row_bytes, row_bytes,
              cudaMemcpyDeviceToHost);

    result.next_real_candidate =
        argmax(multi_logits_host_.data() + vocab_size_, static_cast<std::size_t>(vocab_size_));

    return result;
}

} // namespace trtmc
