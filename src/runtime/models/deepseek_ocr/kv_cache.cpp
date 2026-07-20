/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/deepseek_ocr/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace trtmc {

namespace {

constexpr int32_t kRuntimeBucketRows = 32;

int32_t round_up_rows(int32_t value, int32_t bucket, int32_t maximum) {
    if (bucket <= 1)
        return std::min(std::max(value, 1), maximum);
    const int32_t rounded = ((std::max(value, 1) + bucket - 1) / bucket) * bucket;
    return std::min(rounded, maximum);
}

} // namespace

DeepseekOcrKvCache::DeepseekOcrKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                                       cudaStream_t stream, DType cache_dtype,
                                       DeepseekOcrKvCacheNames names)
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

void DeepseekOcrKvCache::build_attention_mask(std::vector<float>& mask) const {
    // DEPRECATED: use prepare_step() instead.
    const auto width = static_cast<std::size_t>(max_length_) + 1;
    mask.assign(width, kMaskedScore);
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    for (int32_t i = 0; i < valid; ++i)
        mask[static_cast<std::size_t>(i)] = 0.0f;
    mask.back() = 0.0f;
}

int32_t DeepseekOcrKvCache::preferred_cache_rows() const {
    if (!dynamic_binding_enabled_)
        return max_length_;
    return round_up_rows(std::max(position_, 1), kRuntimeBucketRows, max_length_);
}

void DeepseekOcrKvCache::rebind_cache_rows(int32_t cache_rows) {
    if (!dynamic_binding_enabled_ || bound_module_ == nullptr || cache_rows == bound_cache_rows_)
        return;
    const std::vector<int64_t> cache_shape{cache_rows, kv_dim_};
    for (int32_t i = 0; i < num_layers_; ++i) {
        const auto li = static_cast<std::size_t>(i);
        bound_module_->bind_external(names_.cache_k[li], cache_k_[li].data(), cache_shape);
        bound_module_->bind_external(names_.cache_v[li], cache_v_[li].data(), cache_shape);
    }
    bound_cache_rows_ = cache_rows;
}

// Match the engine-declared rank for attention_mask. Different engine families
// wire the causal mask with different shapes:
//   * static decoder (cache-full, e.g. legacy builds): [max_length + 1]
//   * dynamic decoder (standard + triattention):       [1, mask_width]
//   * 3-D decoder mask with query dim:                 [1, 1, mask_width]
// The tensor content is identical (width = current mask_width); only the
// leading broadcast dimensions change.
std::vector<int64_t> DeepseekOcrKvCache::mask_shape_for_engine(int32_t mask_width,
                                                               std::size_t mask_buf_size) const {
    const int32_t mask_rank =
        bound_module_ != nullptr ? bound_module_->input_rank(names_.attention_mask) : 0;
    if (mask_rank == 3)
        return {1, 1, mask_width};
    if (mask_rank == 2 || (mask_rank == 0 && dynamic_binding_enabled_))
        return {1, mask_width};
    return {static_cast<int64_t>(mask_buf_size)};
}

void DeepseekOcrKvCache::write_position_input(TensorMap& inputs, int32_t seq_len) {
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

void DeepseekOcrKvCache::write_batched_mask(TensorMap& inputs, int32_t seq_len) {
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

void DeepseekOcrKvCache::write_bidirectional_mask(TensorMap& inputs, int32_t seq_len) {
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

void DeepseekOcrKvCache::write_decode_mask(TensorMap& inputs) {
    const int32_t valid = std::max(0, std::min(position_, max_length_));
    const int32_t cache_rows = dynamic_binding_enabled_ ? preferred_cache_rows() : max_length_;
    const int32_t mask_width = dynamic_binding_enabled_ ? (cache_rows + 1) : (max_length_ + 1);
    rebind_cache_rows(cache_rows);

    if (mask_buf_.size() < static_cast<std::size_t>(mask_width))
        mask_buf_.assign(static_cast<std::size_t>(mask_width), kMaskedScore);
    std::fill(mask_buf_.begin(), mask_buf_.begin() + mask_width, kMaskedScore);
    for (int32_t i = 0; i < valid; ++i)
        mask_buf_[static_cast<std::size_t>(i)] = 0.0f;
    mask_buf_[static_cast<std::size_t>(mask_width - 1)] = 0.0f;

    Tensor mask_t;
    mask_t.data = mask_buf_.data();
    mask_t.shape = mask_shape_for_engine(mask_width, mask_buf_.size());
    mask_t.dtype = DType::kFloat32;
    inputs[names_.attention_mask] = mask_t;
}

void DeepseekOcrKvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        seq_len = 1;
    write_position_input(inputs, seq_len);
    if (seq_len > 1)
        write_batched_mask(inputs, seq_len);
    else
        write_decode_mask(inputs);
}

void DeepseekOcrKvCache::prepare_bidirectional_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        seq_len = 1;
    write_position_input(inputs, seq_len);
    write_bidirectional_mask(inputs, seq_len);
}

void DeepseekOcrKvCache::bind_to(TrtModule& module) {
    bound_module_ = &module;
    has_position_input_ = module.has_input(names_.position_id);
    // Enable dynamic row binding only when cache_k[0] itself is dynamic.
    // Static-shape engines with fixed [max_length, kv_dim] cache reject
    // setInputShape on cache inputs even when other inputs are dynamic.
    dynamic_binding_enabled_ =
        !names_.cache_k.empty() && module.input_is_dynamic(names_.cache_k.front());
    bound_cache_rows_ = 0;
    const int32_t initial_cache_rows =
        dynamic_binding_enabled_ ? preferred_cache_rows() : max_length_;
    const std::vector<int64_t> cache_shape{initial_cache_rows, kv_dim_};

    for (int32_t i = 0; i < num_layers_; ++i) {
        auto li = static_cast<std::size_t>(i);
        if (dynamic_binding_enabled_) {
            module.bind_external(names_.cache_k[li], cache_k_[li].data(), cache_shape);
            module.bind_external(names_.cache_v[li], cache_v_[li].data(), cache_shape);
            bound_cache_rows_ = initial_cache_rows;
        } else {
            module.bind_external(names_.cache_k[li], cache_k_[li].data());
            module.bind_external(names_.cache_v[li], cache_v_[li].data());
        }
        module.bind_external(names_.present_k[li], present_k_[li].data());
        module.bind_external(names_.present_v[li], present_v_[li].data());
    }
}

void DeepseekOcrKvCache::bind_cache_inputs(TrtModule& module) {
    has_position_input_ = module.has_input(names_.position_id);
    for (int32_t i = 0; i < num_layers_; ++i) {
        auto li = static_cast<std::size_t>(i);
        module.bind_external(names_.cache_k[li], cache_k_[li].data());
        module.bind_external(names_.cache_v[li], cache_v_[li].data());
    }
}

void DeepseekOcrKvCache::write_prefill_kv(const std::vector<const void*>& prefill_k,
                                          const std::vector<const void*>& prefill_v,
                                          int32_t seq_len) {
    if (seq_len <= 0)
        return;
    if (seq_len > max_length_)
        throw std::runtime_error(
            "DeepseekOcrKvCache::write_prefill_kv: seq_len exceeds max_length");
    if (static_cast<int32_t>(prefill_k.size()) != num_layers_ ||
        static_cast<int32_t>(prefill_v.size()) != num_layers_) {
        throw std::runtime_error(
            "DeepseekOcrKvCache::write_prefill_kv: per-layer pointer count mismatch");
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

void DeepseekOcrKvCache::append_prefill_kv(const std::vector<const void*>& prefill_k,
                                           const std::vector<const void*>& prefill_v,
                                           int32_t seq_len) {
    if (seq_len <= 0)
        return;
    if (position_ + seq_len > max_length_)
        throw std::runtime_error(
            "DeepseekOcrKvCache::append_prefill_kv: append exceeds max_length");
    if (static_cast<int32_t>(prefill_k.size()) != num_layers_ ||
        static_cast<int32_t>(prefill_v.size()) != num_layers_) {
        throw std::runtime_error(
            "DeepseekOcrKvCache::append_prefill_kv: per-layer pointer count mismatch");
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

void DeepseekOcrKvCache::set_position(int32_t position) {
    position_ = std::max(0, std::min(position, max_length_));
}

void DeepseekOcrKvCache::advance(int32_t n_tokens) {
    // For now, only single-token advance is supported.
    // n_tokens > 1 reserved for future batched prefill (TASK-10).
    assert(n_tokens == 1 && "DeepseekOcrKvCache::advance: only n_tokens==1 supported");
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

void DeepseekOcrKvCache::reset() {
    // Reset only the logical sequence length. Attention masks hide every
    // stale cache row, and each present row is overwritten before use.
    position_ = 0;
}

std::size_t DeepseekOcrKvCache::device_memory_bytes() const {
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

bool DeepseekOcrKvCache::ok() const {
    if (cache_k_.size() != static_cast<std::size_t>(num_layers_))
        return false;
    for (const auto& t : cache_k_) {
        if (!t.ok())
            return false;
    }
    return true;
}

} // namespace trtmc
