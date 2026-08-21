/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/thinker_kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace trtmc {

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
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    std::fill(mask_buf_.begin(), mask_buf_.end(), kMaskedScore);
    for (int32_t i = 0; i < valid; ++i)
        mask_buf_[static_cast<std::size_t>(i)] = 0.0f;
    mask_buf_.back() = 0.0f;

    Tensor mask_t;
    mask_t.data = mask_buf_.data();
    mask_t.shape = {1, static_cast<int64_t>(mask_buf_.size())};
    mask_t.dtype = DType::kFloat32;
    inputs[names_.attention_mask] = mask_t;
}

void VoiceChatThinkerKvCache::bind_to(TrtModule& module) {
    for (int32_t i = 0; i < num_layers_; ++i) {
        const auto layer = static_cast<std::size_t>(i);
        module.bind_external(names_.cache_k[layer], cache_k_[layer].data());
        module.bind_external(names_.cache_v[layer], cache_v_[layer].data());
        module.bind_external(names_.present_k[layer], present_k_[layer].data());
        module.bind_external(names_.present_v[layer], present_v_[layer].data());
    }
}

void VoiceChatThinkerKvCache::advance() {
    // Copy present K/V (single row) into cache at current position.
    // present_k_[layer] is [1, kv_dim] → copy to cache_k_[layer][position_, :]
    auto row_bytes = static_cast<std::size_t>(kv_dim_) * sizeof(float);

    if (position_ < max_length_) {
        // Normal append: write to position_ slot
        auto offset = static_cast<std::size_t>(position_) * row_bytes;
        for (int32_t i = 0; i < num_layers_; ++i) {
            auto li = static_cast<std::size_t>(i);
            cudaMemcpyAsync(static_cast<uint8_t*>(cache_k_[li].data()) + offset,
                            present_k_[li].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(static_cast<uint8_t*>(cache_v_[li].data()) + offset,
                            present_v_[li].data(), row_bytes, cudaMemcpyDeviceToDevice, stream_);
        }
        ++position_;
    } else {
        // Cache full: shift [1..max) → [0..max-1), then write at tail
        auto shift_bytes = static_cast<std::size_t>(max_length_ - 1) * row_bytes;
        auto tail_offset = shift_bytes;
        for (int32_t i = 0; i < num_layers_; ++i) {
            auto li = static_cast<std::size_t>(i);
            auto* ck = static_cast<uint8_t*>(cache_k_[li].data());
            auto* cv = static_cast<uint8_t*>(cache_v_[li].data());
            cudaMemcpyAsync(ck, ck + row_bytes, shift_bytes, cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(cv, cv + row_bytes, shift_bytes, cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(ck + tail_offset, present_k_[li].data(), row_bytes,
                            cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(cv + tail_offset, present_v_[li].data(), row_bytes,
                            cudaMemcpyDeviceToDevice, stream_);
        }
        // position_ stays at max_length_ (cache is full, all slots visible)
    }
}

void VoiceChatThinkerKvCache::reset() {
    // Reset only the logical sequence length. Attention masks hide every
    // stale cache row, and each present row is overwritten before use.
    position_ = 0;
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
