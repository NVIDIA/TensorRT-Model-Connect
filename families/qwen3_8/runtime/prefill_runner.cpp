/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_8/runtime/prefill_runner.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace trtmc {

namespace {
constexpr float kMaskedScore = -1.0e4F;

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen38PrefillRunner: ") + what + ": " +
                                 cudaGetErrorString(status));
}
} // namespace

Qwen38PrefillRunner::Qwen38PrefillRunner(std::unique_ptr<ITrtModule> module,
                                         Qwen38HybridState& state, int32_t chunk_len,
                                         int32_t hidden_size, int32_t vocab_size,
                                         cudaStream_t stream)
    : module_(std::move(module)), state_(state), stream_(stream), chunk_len_(chunk_len),
      hidden_size_(hidden_size), vocab_size_(vocab_size) {
    if (!module_ || !module_->ok())
        throw std::runtime_error("Qwen38PrefillRunner: invalid prefill module");
    if (chunk_len_ < 2)
        throw std::runtime_error("Qwen38PrefillRunner: chunk_len must be >= 2");

    const Qwen38KvCache& kv = *state_.kv_cache();
    present_k_.reserve(static_cast<std::size_t>(kv.num_layers()));
    present_v_.reserve(static_cast<std::size_t>(kv.num_layers()));
    for (int32_t i = 0; i < kv.num_layers(); ++i) {
        present_k_.emplace_back(std::vector<int64_t>{chunk_len_, kv.kv_dim()}, kv.dtype(), stream_);
        present_v_.emplace_back(std::vector<int64_t>{chunk_len_, kv.kv_dim()}, kv.dtype(), stream_);
        if (!present_k_.back().ok() || !present_v_.back().ok())
            throw std::runtime_error("Qwen38PrefillRunner: failed to allocate present K/V scratch");
    }
    tokens_buf_.resize(static_cast<std::size_t>(chunk_len_));
    positions_buf_.resize(static_cast<std::size_t>(chunk_len_));
}

void Qwen38PrefillRunner::bind_state() {
    Qwen38KvCache& kv = *state_.kv_cache();
    Qwen38RecurrentState& ssm = *state_.recurrent_state();

    for (int32_t i = 0; i < kv.num_layers(); ++i) {
        const auto suffix = "_" + std::to_string(i);
        const auto li = static_cast<std::size_t>(i);
        module_->bind_external("cache_k" + suffix, kv.cache_k(i).data());
        module_->bind_external("cache_v" + suffix, kv.cache_v(i).data());
        module_->bind_external("present_k" + suffix, present_k_[li].data());
        module_->bind_external("present_v" + suffix, present_v_[li].data());
    }
    for (int32_t i = 0; i < ssm.num_layers(); ++i) {
        const auto suffix = "_" + std::to_string(i);
        module_->bind_external("conv_state" + suffix, ssm.state_ptr(0, i));
        module_->bind_external("ssm_state" + suffix, ssm.state_ptr(1, i));
        module_->bind_external("present_conv" + suffix, ssm.present_ptr(0, i));
        module_->bind_external("present_ssm" + suffix, ssm.present_ptr(1, i));
    }
    bound_ = true;
}

bool Qwen38PrefillRunner::fits(int32_t n) const {
    return state_.kv_cache()->position() + n <= state_.kv_cache()->max_length();
}

void Qwen38PrefillRunner::run_chunk(const int32_t* tokens, int32_t n, std::vector<float>& logits,
                                    std::vector<float>* hidden_rows) {
    if (!bound_)
        throw std::runtime_error("Qwen38PrefillRunner: bind_state() must be called first");
    if (n < 1 || n > chunk_len_)
        throw std::runtime_error("Qwen38PrefillRunner::run_chunk: n out of range");
    Qwen38KvCache& kv = *state_.kv_cache();
    Qwen38RecurrentState& ssm = *state_.recurrent_state();
    if (!fits(n))
        throw std::runtime_error("Qwen38PrefillRunner::run_chunk: chunk exceeds KV cache length");

    const int32_t pos0 = kv.position();
    const int32_t max_len = kv.max_length();
    const int32_t kv_len = max_len + chunk_len_;

    // Rows >= n are padding: any token id / position works because the engine
    // keeps them out of the recurrent state and the runtime never commits
    // their K/V rows. Pad with the last real token and consecutive positions.
    for (int32_t i = 0; i < chunk_len_; ++i) {
        tokens_buf_[static_cast<std::size_t>(i)] = tokens[std::min(i, n - 1)];
        positions_buf_[static_cast<std::size_t>(i)] = pos0 + i;
    }

    // Additive mask (chunk_len, max_len + chunk_len): visible cache prefix
    // [0, pos0), then causal over the chunk's own rows.
    mask_buf_.assign(static_cast<std::size_t>(chunk_len_) * static_cast<std::size_t>(kv_len),
                     kMaskedScore);
    for (int32_t i = 0; i < chunk_len_; ++i) {
        float* row =
            mask_buf_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(kv_len);
        std::fill(row, row + pos0, 0.0F);
        std::fill(row + max_len, row + max_len + i + 1, 0.0F);
    }

    int32_t valid_len = n;
    TensorMap inputs;
    {
        Tensor t;
        t.data = tokens_buf_.data();
        t.shape = {chunk_len_};
        t.dtype = DType::kInt32;
        inputs["token_ids"] = t;
        t.data = positions_buf_.data();
        inputs["position_id"] = t;
        t.data = &valid_len;
        t.shape = {1};
        inputs["valid_len"] = t;
        t.data = mask_buf_.data();
        t.shape = {chunk_len_, kv_len};
        t.dtype = DType::kFloat32;
        inputs["attention_mask"] = t;
    }

    module_->forward_async(inputs);
    module_->sync();

    if (logits_ptr_ == nullptr) {
        logits_ptr_ = module_->device_ptr("logits");
        if (logits_ptr_ == nullptr)
            throw std::runtime_error("Qwen38PrefillRunner: prefill module has no 'logits' output");
    }
    logits.resize(static_cast<std::size_t>(vocab_size_));
    check_cuda(cudaMemcpy(logits.data(), logits_ptr_, logits.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "failed to copy logits");

    if (hidden_rows != nullptr) {
        if (hidden_ptr_ == nullptr) {
            hidden_ptr_ = module_->device_ptr("hidden_states");
            if (hidden_ptr_ == nullptr)
                throw std::runtime_error(
                    "Qwen38PrefillRunner: prefill module has no 'hidden_states' output");
        }
        hidden_rows->resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(hidden_size_));
        check_cuda(cudaMemcpy(hidden_rows->data(), hidden_ptr_, hidden_rows->size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "failed to copy hidden states");
    }

    std::vector<const void*> pk;
    std::vector<const void*> pv;
    pk.reserve(present_k_.size());
    pv.reserve(present_v_.size());
    for (std::size_t i = 0; i < present_k_.size(); ++i) {
        pk.push_back(present_k_[i].data());
        pv.push_back(present_v_[i].data());
    }
    kv.append_prefill_kv(pk, pv, n);
    ssm.advance(n);
}

} // namespace trtmc
