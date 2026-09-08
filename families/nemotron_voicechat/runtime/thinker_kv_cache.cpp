/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_voicechat/runtime/thinker_kv_cache.h"

#include "families/nemotron_voicechat/runtime/session_state.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

void require_cuda_success(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("VoiceChat thinker ") + operation +
                                 " failed: " + cudaGetErrorString(status));
}

} // namespace

VoiceChatThinkerKvCacheNames::VoiceChatThinkerKvCacheNames(int32_t num_layers) {
    cache_k.reserve(static_cast<std::size_t>(num_layers));
    cache_v.reserve(static_cast<std::size_t>(num_layers));
    present_k.reserve(static_cast<std::size_t>(num_layers));
    present_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        const auto suffix = "_" + std::to_string(i);
        cache_k.push_back("cache_k" + suffix);
        cache_v.push_back("cache_v" + suffix);
        present_k.push_back("present_k" + suffix);
        present_v.push_back("present_v" + suffix);
    }
}

VoiceChatThinkerKvCache::VoiceChatThinkerKvCache(int32_t num_layers, int32_t max_length,
                                                 int32_t kv_dim, cudaStream_t stream)
    : names_(num_layers), num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim),
      stream_(stream) {

    cache_k_.reserve(static_cast<std::size_t>(num_layers));
    cache_v_.reserve(static_cast<std::size_t>(num_layers));
    present_k_.reserve(static_cast<std::size_t>(num_layers));
    present_v_.reserve(static_cast<std::size_t>(num_layers));

    for (int32_t i = 0; i < num_layers; ++i) {
        cache_k_.emplace_back(std::vector<int64_t>{max_length, kv_dim}, DType::kFloat32, stream);
        cache_v_.emplace_back(std::vector<int64_t>{max_length, kv_dim}, DType::kFloat32, stream);
        present_k_.emplace_back(std::vector<int64_t>{1, kv_dim}, DType::kFloat32, stream);
        present_v_.emplace_back(std::vector<int64_t>{1, kv_dim}, DType::kFloat32, stream);
    }

    mask_buf_.resize(static_cast<std::size_t>(max_length) + 1);

    reset();
}

// Masked score constant is model-local.
static constexpr float kMaskedScore = -1.0e4F;

void VoiceChatThinkerKvCache::prepare_step(TensorMap& inputs) {
    const auto cache = nemotron_voicechat::rolling_cache_position(logical_position_, max_length_,
                                                                  pinned_prefix_rows_);
    std::fill(mask_buf_.begin(), mask_buf_.end(), kMaskedScore);
    for (int32_t i = 0; i < cache.valid_rows; ++i)
        mask_buf_[static_cast<std::size_t>(i)] = 0.0f;
    mask_buf_.back() = 0.0f;

    Tensor mask_t;
    mask_t.data = mask_buf_.data();
    mask_t.shape = {1, static_cast<int64_t>(mask_buf_.size())};
    mask_t.dtype = DType::kFloat32;
    inputs[names_.attention_mask] = mask_t;
}

void VoiceChatThinkerKvCache::bind_to(ITrtModule& module) {
    for (int32_t i = 0; i < num_layers_; ++i) {
        const auto layer = static_cast<std::size_t>(i);
        module.bind_external(names_.cache_k[layer], cache_k_[layer].data());
        module.bind_external(names_.cache_v[layer], cache_v_[layer].data());
        module.bind_external(names_.present_k[layer], present_k_[layer].data());
        module.bind_external(names_.present_v[layer], present_v_[layer].data());
    }
}

void VoiceChatThinkerKvCache::advance() {
    // Attention is position-free in the VoiceChat thinker, so the joint K/V
    // row permutation of a ring is semantically invisible. A one-row ring copy
    // also avoids the undefined overlapping device memcpy used by the prior
    // full-cache shift.
    const auto cache = nemotron_voicechat::rolling_cache_position(logical_position_, max_length_,
                                                                  pinned_prefix_rows_);
    const auto row_bytes = static_cast<std::size_t>(kv_dim_) * sizeof(float);
    const auto offset = static_cast<std::size_t>(cache.write_row) * row_bytes;
    for (int32_t i = 0; i < num_layers_; ++i) {
        const auto layer = static_cast<std::size_t>(i);
        require_cuda_success(cudaMemcpyAsync(static_cast<uint8_t*>(cache_k_[layer].data()) + offset,
                                             present_k_[layer].data(), row_bytes,
                                             cudaMemcpyDeviceToDevice, stream_),
                             "VoiceChat thinker K-cache append");
        require_cuda_success(cudaMemcpyAsync(static_cast<uint8_t*>(cache_v_[layer].data()) + offset,
                                             present_v_[layer].data(), row_bytes,
                                             cudaMemcpyDeviceToDevice, stream_),
                             "VoiceChat thinker V-cache append");
    }
    ++logical_position_;
}

void VoiceChatThinkerKvCache::pin_current_prefix() {
    if (pinned_prefix_rows_ != 0)
        throw std::logic_error("VoiceChat thinker KV prefix is already pinned");
    if (logical_position_ <= 0 || logical_position_ >= max_length_)
        throw std::runtime_error(
            "VoiceChat thinker system prompt must leave room for rolling cache rows");
    pinned_prefix_rows_ = static_cast<int32_t>(logical_position_);
}

void VoiceChatThinkerKvCache::capture_prompt_snapshot() {
    if (prompt_snapshot_ready_)
        throw std::logic_error("VoiceChat thinker KV prompt snapshot is already captured");
    if (pinned_prefix_rows_ <= 0 || logical_position_ != pinned_prefix_rows_)
        throw std::logic_error(
            "VoiceChat thinker KV prompt snapshot requires an exact pinned prefix");

    std::vector<DeviceTensor> snapshot_k;
    std::vector<DeviceTensor> snapshot_v;
    snapshot_k.reserve(static_cast<std::size_t>(num_layers_));
    snapshot_v.reserve(static_cast<std::size_t>(num_layers_));
    const auto shape = std::vector<int64_t>{pinned_prefix_rows_, kv_dim_};
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        snapshot_k.emplace_back(shape, DType::kFloat32, stream_);
        snapshot_v.emplace_back(shape, DType::kFloat32, stream_);
        if (!snapshot_k.back().ok() || !snapshot_v.back().ok())
            throw std::runtime_error("VoiceChat failed to allocate thinker KV prompt snapshot");
    }

    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        const auto bytes = snapshot_k[index].nbytes();
        require_cuda_success(cudaMemcpyAsync(snapshot_k[index].data(), cache_k_[index].data(),
                                             bytes, cudaMemcpyDeviceToDevice, stream_),
                             "KV prompt K-cache capture");
        require_cuda_success(cudaMemcpyAsync(snapshot_v[index].data(), cache_v_[index].data(),
                                             bytes, cudaMemcpyDeviceToDevice, stream_),
                             "KV prompt V-cache capture");
    }
    require_cuda_success(cudaStreamSynchronize(stream_), "KV prompt snapshot sync");
    prompt_snapshot_k_ = std::move(snapshot_k);
    prompt_snapshot_v_ = std::move(snapshot_v);
    prompt_snapshot_rows_ = pinned_prefix_rows_;
    prompt_snapshot_ready_ = true;
}

void VoiceChatThinkerKvCache::restore_prompt_snapshot() {
    if (!prompt_snapshot_ready_ || prompt_snapshot_rows_ <= 0)
        throw std::logic_error("VoiceChat thinker KV prompt snapshot is unavailable");
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        const auto bytes = prompt_snapshot_k_[index].nbytes();
        require_cuda_success(cudaMemcpyAsync(cache_k_[index].data(),
                                             prompt_snapshot_k_[index].data(), bytes,
                                             cudaMemcpyDeviceToDevice, stream_),
                             "KV prompt K-cache restore");
        require_cuda_success(cudaMemcpyAsync(cache_v_[index].data(),
                                             prompt_snapshot_v_[index].data(), bytes,
                                             cudaMemcpyDeviceToDevice, stream_),
                             "KV prompt V-cache restore");
    }
    require_cuda_success(cudaStreamSynchronize(stream_), "KV prompt restore sync");
    logical_position_ = prompt_snapshot_rows_;
    pinned_prefix_rows_ = prompt_snapshot_rows_;
}

void VoiceChatThinkerKvCache::reset() {
    // Reset only the logical sequence length. Attention masks hide every
    // stale cache row, and each present row is overwritten before use.
    logical_position_ = 0;
    pinned_prefix_rows_ = 0;
}

bool VoiceChatThinkerKvCache::ok() const {
    if (cache_k_.size() != static_cast<std::size_t>(num_layers_))
        return false;
    for (const auto& t : cache_k_) {
        if (!t.ok())
            return false;
    }
    return true;
}

} // namespace trtmc
