/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/olmo/runtime/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace trtmc {

OlmoKvCache::OlmoKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                         cudaStream_t stream, DType cache_dtype, OlmoKvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim), stream_(stream),
      cache_dtype_(cache_dtype), cache_element_size_(dtype_size(cache_dtype)),
      names_(std::move(names)) {

    // If names were not supplied, generate standard defaults.
    if (names_.cache_k.empty()) {
        names_.cache_k.reserve(static_cast<std::size_t>(num_layers));
        names_.cache_v.reserve(static_cast<std::size_t>(num_layers));
        names_.present_k.reserve(static_cast<std::size_t>(num_layers));
        names_.present_v.reserve(static_cast<std::size_t>(num_layers));
        for (int32_t i = 0; i < num_layers; ++i) {
            std::string suffix = "_" + std::to_string(i);
            names_.cache_k.push_back("cache_k" + suffix);
            names_.cache_v.push_back("cache_v" + suffix);
            names_.present_k.push_back("present_k" + suffix);
            names_.present_v.push_back("present_v" + suffix);
        }
    }

    cache_k_.reserve(static_cast<std::size_t>(num_layers));
    cache_v_.reserve(static_cast<std::size_t>(num_layers));
    present_k_.reserve(static_cast<std::size_t>(num_layers));
    present_v_.reserve(static_cast<std::size_t>(num_layers));

    for (int32_t i = 0; i < num_layers; ++i) {
        cache_k_.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype_, stream);
        cache_v_.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype_, stream);
        present_k_.emplace_back(std::vector<int64_t>{1, kv_dim}, cache_dtype_, stream);
        present_v_.emplace_back(std::vector<int64_t>{1, kv_dim}, cache_dtype_, stream);
    }

    // Pre-allocate mask buffer: [max_length + 1] for dense causal mask.
    mask_buf_.resize(static_cast<std::size_t>(max_length) + 1);

    reset();
}

// Masked score constant is model-local.
static constexpr float kMaskedScore = -1.0e4F;

void OlmoKvCache::write_position_input(TensorMap& inputs, int32_t seq_len) {
    if (!has_position_input_)
        return;
    pos_buf_vec_.resize(static_cast<std::size_t>(seq_len));
    for (int32_t i = 0; i < seq_len; ++i)
        pos_buf_vec_[static_cast<std::size_t>(i)] = position_ + i;
    Tensor pos_t;
    pos_t.data = pos_buf_vec_.data();
    pos_t.shape = {static_cast<int64_t>(seq_len)};
    pos_t.dtype = DType::kInt32;
    inputs[names_.position_id] = pos_t;
}

void OlmoKvCache::write_batched_mask(TensorMap& inputs, int32_t seq_len) {
    // Batched prefill mask: (seq_len, max_length + seq_len). Columns
    // [0, valid) are visible cache, [valid, max_length) are stale slots,
    // [max_length, max_length+seq_len) are the new tokens — causal so
    // token i sees tokens 0..i.
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    const int32_t kv_len = max_length_ + seq_len;
    const std::size_t total = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(kv_len);
    mask_buf_.assign(total, kMaskedScore);
    for (int32_t i = 0; i < seq_len; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * static_cast<std::size_t>(kv_len);
        for (int32_t j = 0; j < valid; ++j)
            mask_buf_[row + static_cast<std::size_t>(j)] = 0.0f;
        for (int32_t j = 0; j <= i; ++j)
            mask_buf_[row + static_cast<std::size_t>(max_length_) + static_cast<std::size_t>(j)] =
                0.0f;
    }
    Tensor mask_t;
    mask_t.data = mask_buf_.data();
    mask_t.shape = {static_cast<int64_t>(seq_len), static_cast<int64_t>(kv_len)};
    mask_t.dtype = DType::kFloat32;
    inputs[names_.attention_mask] = mask_t;
}

void OlmoKvCache::write_bidirectional_mask(TensorMap& inputs, int32_t seq_len) {
    // Diffusion block mask: all valid prefix cache rows are visible, stale cache
    // rows are hidden, and every token in the current block can see every other
    // token in the current block.
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    const int32_t kv_len = max_length_ + seq_len;
    const std::size_t total = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(kv_len);
    mask_buf_.assign(total, kMaskedScore);
    for (int32_t i = 0; i < seq_len; ++i) {
        const std::size_t row = static_cast<std::size_t>(i) * static_cast<std::size_t>(kv_len);
        for (int32_t j = 0; j < valid; ++j)
            mask_buf_[row + static_cast<std::size_t>(j)] = 0.0f;
        for (int32_t j = 0; j < seq_len; ++j)
            mask_buf_[row + static_cast<std::size_t>(max_length_) + static_cast<std::size_t>(j)] =
                0.0f;
    }
    Tensor mask_t;
    mask_t.data = mask_buf_.data();
    mask_t.shape = {static_cast<int64_t>(seq_len), static_cast<int64_t>(kv_len)};
    mask_t.dtype = DType::kFloat32;
    inputs[names_.attention_mask] = mask_t;
}

void OlmoKvCache::write_decode_mask(TensorMap& inputs) {
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    const int32_t mask_width = max_length_ + 1;
    if (mask_buf_.size() != static_cast<std::size_t>(mask_width))
        mask_buf_.resize(static_cast<std::size_t>(mask_width));
    std::fill(mask_buf_.begin(), mask_buf_.end(), kMaskedScore);
    for (int32_t i = 0; i < valid; ++i)
        mask_buf_[static_cast<std::size_t>(i)] = 0.0F;
    mask_buf_.back() = 0.0F;

    Tensor mask_t;
    mask_t.data = mask_buf_.data();
    mask_t.shape = {1, mask_width};
    mask_t.dtype = DType::kFloat32;
    inputs[names_.attention_mask] = mask_t;
}

void OlmoKvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        seq_len = 1;
    write_position_input(inputs, seq_len);
    if (seq_len > 1)
        write_batched_mask(inputs, seq_len);
    else
        write_decode_mask(inputs);
}

void OlmoKvCache::prepare_bidirectional_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        seq_len = 1;
    write_position_input(inputs, seq_len);
    write_bidirectional_mask(inputs, seq_len);
}

void OlmoKvCache::bind_to(ITrtModule& module) {
    if (names_.cache_k.empty())
        throw std::runtime_error("olmo KV cache has no cache tensor names");
    if (!module.has_input(names_.attention_mask) ||
        module.tensor_shape(names_.attention_mask) != std::vector<int64_t>{1, max_length_ + 1}) {
        throw std::runtime_error("olmo decoder has an invalid attention-mask contract");
    }

    const std::vector<int64_t> cache_shape{max_length_, kv_dim_};
    const std::vector<int64_t> present_shape{1, kv_dim_};
    bound_module_ = &module;
    has_position_input_ = module.has_input(names_.position_id);
    for (int32_t i = 0; i < num_layers_; ++i) {
        const auto layer = static_cast<std::size_t>(i);
        if (!module.has_input(names_.cache_k[layer]) || !module.has_input(names_.cache_v[layer]) ||
            !module.has_output(names_.present_k[layer]) ||
            !module.has_output(names_.present_v[layer]) ||
            module.tensor_shape(names_.cache_k[layer]) != cache_shape ||
            module.tensor_shape(names_.cache_v[layer]) != cache_shape ||
            module.tensor_shape(names_.present_k[layer]) != present_shape ||
            module.tensor_shape(names_.present_v[layer]) != present_shape) {
            throw std::runtime_error("olmo decoder does not match its fixed KV contract");
        }
        module.bind_external(names_.cache_k[layer], cache_k_[layer].data());
        module.bind_external(names_.cache_v[layer], cache_v_[layer].data());
        module.bind_external(names_.present_k[layer], present_k_[layer].data());
        module.bind_external(names_.present_v[layer], present_v_[layer].data());
    }
}

void OlmoKvCache::bind_cache_inputs(ITrtModule& module) {
    has_position_input_ = module.has_input(names_.position_id);
    for (int32_t i = 0; i < num_layers_; ++i) {
        auto li = static_cast<std::size_t>(i);
        module.bind_external(names_.cache_k[li], cache_k_[li].data());
        module.bind_external(names_.cache_v[li], cache_v_[li].data());
    }
}

void OlmoKvCache::write_prefill_kv(const std::vector<const void*>& prefill_k,
                                   const std::vector<const void*>& prefill_v, int32_t seq_len) {
    if (seq_len <= 0)
        return;
    if (seq_len > max_length_)
        throw std::runtime_error("OlmoKvCache::write_prefill_kv: seq_len exceeds max_length");
    if (static_cast<int32_t>(prefill_k.size()) != num_layers_ ||
        static_cast<int32_t>(prefill_v.size()) != num_layers_) {
        throw std::runtime_error("OlmoKvCache::write_prefill_kv: per-layer pointer count mismatch");
    }
    const auto row_bytes = static_cast<std::size_t>(kv_dim_) * cache_element_size_;
    const auto block_bytes = static_cast<std::size_t>(seq_len) * row_bytes;
    for (int32_t i = 0; i < num_layers_; ++i) {
        auto li = static_cast<std::size_t>(i);
        cudaMemcpyAsync(cache_k_[li].data(), prefill_k[li], block_bytes, cudaMemcpyDeviceToDevice,
                        stream_);
        cudaMemcpyAsync(cache_v_[li].data(), prefill_v[li], block_bytes, cudaMemcpyDeviceToDevice,
                        stream_);
    }
    position_ = seq_len;
}

void OlmoKvCache::append_prefill_kv(const std::vector<const void*>& prefill_k,
                                    const std::vector<const void*>& prefill_v, int32_t seq_len) {
    if (seq_len <= 0)
        return;
    if (position_ + seq_len > max_length_)
        throw std::runtime_error("OlmoKvCache::append_prefill_kv: append exceeds max_length");
    if (static_cast<int32_t>(prefill_k.size()) != num_layers_ ||
        static_cast<int32_t>(prefill_v.size()) != num_layers_) {
        throw std::runtime_error(
            "OlmoKvCache::append_prefill_kv: per-layer pointer count mismatch");
    }
    const auto row_bytes = static_cast<std::size_t>(kv_dim_) * cache_element_size_;
    const auto block_bytes = static_cast<std::size_t>(seq_len) * row_bytes;
    const auto offset = static_cast<std::size_t>(position_) * row_bytes;
    for (int32_t i = 0; i < num_layers_; ++i) {
        auto li = static_cast<std::size_t>(i);
        cudaMemcpyAsync(static_cast<uint8_t*>(cache_k_[li].data()) + offset, prefill_k[li],
                        block_bytes, cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(static_cast<uint8_t*>(cache_v_[li].data()) + offset, prefill_v[li],
                        block_bytes, cudaMemcpyDeviceToDevice, stream_);
    }
    position_ += seq_len;
}

void OlmoKvCache::set_position(int32_t position) {
    position_ = std::max(0, std::min(position, max_length_));
}

void OlmoKvCache::advance(int32_t n_tokens) {
    // For now, only single-token advance is supported.
    // n_tokens > 1 reserved for future batched prefill (TASK-10).
    assert(n_tokens == 1 && "OlmoKvCache::advance: only n_tokens==1 supported");
    (void)n_tokens;

    // Copy present K/V (single row) into cache at current position.
    // present_k_[layer] is [1, kv_dim] → copy to cache_k_[layer][position_, :]
    auto row_bytes = static_cast<std::size_t>(kv_dim_) * cache_element_size_;

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

void OlmoKvCache::reset() {
    // Reset only the logical sequence length. Attention masks hide every
    // stale cache row, and each present row is overwritten before use.
    position_ = 0;
}

std::size_t OlmoKvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& t : cache_k_)
        total += t.nbytes();
    for (const auto& t : cache_v_)
        total += t.nbytes();
    for (const auto& t : present_k_)
        total += t.nbytes();
    for (const auto& t : present_v_)
        total += t.nbytes();
    return total;
}

bool OlmoKvCache::ok() const {
    if (cache_k_.size() != static_cast<std::size_t>(num_layers_))
        return false;
    for (const auto& t : cache_k_) {
        if (!t.ok())
            return false;
    }
    return true;
}

} // namespace trtmc
