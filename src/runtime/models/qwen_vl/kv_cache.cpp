/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen_vl/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

void validate_native_scalar_input(TrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("Qwen-VL native KV input '" + name + "' must be int32 [1]");
    }
}

bool valid_native_cache_shape(const std::vector<int64_t>& shape, int32_t max_length,
                              int32_t kv_dim) {
    return shape.size() == 4 && shape[0] == 1 && shape[1] > 0 &&
           shape[2] == static_cast<int64_t>(max_length) && shape[3] > 0 &&
           shape[1] * shape[3] == kv_dim;
}

void validate_native_cache_pair(TrtModule& module, const std::string& cache_name,
                                const std::string& present_name, int32_t max_length, int32_t kv_dim,
                                DType cache_dtype) {
    if (!module.has_input(cache_name) || !module.has_output(present_name)) {
        throw std::runtime_error("Qwen-VL native KV engine is missing cache/present pair '" +
                                 cache_name + "'/'" + present_name + "'");
    }
    const auto cache_shape = module.tensor_shape(cache_name);
    if (!valid_native_cache_shape(cache_shape, max_length, kv_dim) ||
        module.tensor_shape(present_name) != cache_shape) {
        throw std::runtime_error(
            "Qwen-VL native KV cache/present tensors must share [1,Hkv,capacity,D]");
    }
    if (module.tensor_dtype(cache_name) != cache_dtype ||
        module.tensor_dtype(present_name) != cache_dtype) {
        throw std::runtime_error("Qwen-VL native KV cache dtype does not match model precision");
    }
}

bool all_tensors_ok(const std::vector<DeviceTensor>& tensors) {
    for (const auto& tensor : tensors) {
        if (!tensor.ok())
            return false;
    }
    return true;
}

void populate_default_cache_names(int32_t num_layers, QwenVlKvCacheNames& names) {
    if (!names.cache_k.empty())
        return;

    names.cache_k.reserve(static_cast<std::size_t>(num_layers));
    names.cache_v.reserve(static_cast<std::size_t>(num_layers));
    names.present_k.reserve(static_cast<std::size_t>(num_layers));
    names.present_v.reserve(static_cast<std::size_t>(num_layers));
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        const auto suffix = "_" + std::to_string(layer);
        names.cache_k.push_back("cache_k" + suffix);
        names.cache_v.push_back("cache_v" + suffix);
        names.present_k.push_back("present_k" + suffix);
        names.present_v.push_back("present_v" + suffix);
    }
}

void validate_cache_name_counts(int32_t num_layers, const QwenVlKvCacheNames& names) {
    const auto expected = static_cast<std::size_t>(num_layers);
    if (names.cache_k.size() != expected || names.cache_v.size() != expected ||
        names.present_k.size() != expected || names.present_v.size() != expected) {
        throw std::invalid_argument("QwenVlKvCache per-layer tensor name count mismatch");
    }
}

bool allocate_cache_layers(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                           DType cache_dtype, cudaStream_t stream,
                           std::vector<DeviceTensor>& cache_k, std::vector<DeviceTensor>& cache_v) {
    const auto layer_count = static_cast<std::size_t>(num_layers);
    cache_k.reserve(layer_count);
    cache_v.reserve(layer_count);
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        cache_k.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype, stream);
        if (!cache_k.back().ok())
            return false;
        cache_v.emplace_back(std::vector<int64_t>{max_length, kv_dim}, cache_dtype, stream);
        if (!cache_v.back().ok())
            return false;
    }
    return true;
}

} // namespace

QwenVlKvCache::QwenVlKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                             cudaStream_t stream, DType cache_dtype, QwenVlKvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim), cache_dtype_(cache_dtype),
      names_(std::move(names)) {
    if (num_layers <= 0 || max_length <= 0 || kv_dim <= 0)
        throw std::invalid_argument("QwenVlKvCache dimensions must be positive");

    populate_default_cache_names(num_layers, names_);
    validate_cache_name_counts(num_layers, names_);
    if (!allocate_cache_layers(num_layers, max_length, kv_dim, cache_dtype, stream, cache_k_,
                               cache_v_))
        return;
    reset();
}

void QwenVlKvCache::validate_native_kv_contract(TrtModule& module) const {
    validate_native_scalar_input(module, names_.cache_write_indices);
    validate_native_scalar_input(module, names_.key_value_lengths);
    if (module.has_input("attention_mask")) {
        throw std::runtime_error(
            "Qwen-VL native KV engine must use key_value_lengths, not attention_mask");
    }
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        validate_native_cache_pair(module, names_.cache_k[index], names_.present_k[index],
                                   max_length_, kv_dim_, cache_dtype_);
        validate_native_cache_pair(module, names_.cache_v[index], names_.present_v[index],
                                   max_length_, kv_dim_, cache_dtype_);
    }
}

void QwenVlKvCache::bind_native_cache(TrtModule& module) {
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        module.bind_external(names_.cache_k[index], cache_k_[index].data());
        module.bind_external(names_.cache_v[index], cache_v_[index].data());
        if (module.device_ptr(names_.cache_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.present_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.cache_v[index]) != cache_v_[index].data() ||
            module.device_ptr(names_.present_v[index]) != cache_v_[index].data()) {
            throw std::runtime_error(
                "Qwen-VL native KV engine did not preserve cache/present aliasing");
        }
    }
}

void QwenVlKvCache::bind_to(TrtModule& module) {
    has_position_input_ = module.has_input(names_.position_id);
    has_mrope_position_input_ = module.has_input("mrope_position_ids");
    validate_native_kv_contract(module);
    bind_native_cache(module);
}

void QwenVlKvCache::bind_cache_inputs(TrtModule& module) {
    bind_to(module);
}

void QwenVlKvCache::write_position_input(TensorMap& inputs, int32_t seq_len) {
    if (has_position_input_) {
        pos_buf_vec_.resize(static_cast<std::size_t>(seq_len));
        for (int32_t offset = 0; offset < seq_len; ++offset)
            pos_buf_vec_[static_cast<std::size_t>(offset)] = position_ + offset;
        inputs[names_.position_id] =
            Tensor{pos_buf_vec_.data(), {static_cast<int64_t>(seq_len)}, DType::kInt32};
    }

    // Batched VL prefill supplies model-specific mRoPE positions explicitly.
    if (has_mrope_position_input_ && seq_len == 1) {
        mrope_pos_buf_.fill(position_);
        inputs["mrope_position_ids"] = Tensor{mrope_pos_buf_.data(), {3, 1}, DType::kInt32};
    }
}

void QwenVlKvCache::write_native_kv_inputs(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        throw std::invalid_argument("Qwen-VL native KV step length must be positive");
    if (seq_len > max_length_ - position_)
        throw std::runtime_error("Qwen-VL sequence exceeds fixed KV capacity");
    cache_write_index_ = position_;
    key_value_length_ = position_ + seq_len;
    inputs[names_.cache_write_indices] = Tensor{&cache_write_index_, {1}, DType::kInt32};
    inputs[names_.key_value_lengths] = Tensor{&key_value_length_, {1}, DType::kInt32};
}

void QwenVlKvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        throw std::invalid_argument("Qwen-VL native KV step length must be positive");
    write_position_input(inputs, seq_len);
    write_native_kv_inputs(inputs, seq_len);
}

void QwenVlKvCache::validate_native_aliases(const std::vector<const void*>& present_k,
                                            const std::vector<const void*>& present_v) const {
    if (present_k.size() != static_cast<std::size_t>(num_layers_) ||
        present_v.size() != static_cast<std::size_t>(num_layers_)) {
        throw std::runtime_error("Qwen-VL native KV per-layer pointer count mismatch");
    }
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        if (present_k[index] != cache_k_[index].data() ||
            present_v[index] != cache_v_[index].data()) {
            throw std::runtime_error(
                "Qwen-VL native prefill present tensors must alias the KV cache");
        }
    }
}

void QwenVlKvCache::append_prefill_kv(const std::vector<const void*>& prefill_k,
                                      const std::vector<const void*>& prefill_v, int32_t seq_len) {
    if (seq_len <= 0)
        throw std::invalid_argument("Qwen-VL native prefill length must be positive");
    if (seq_len > max_length_ - position_)
        throw std::runtime_error("Qwen-VL native prefill exceeds fixed KV capacity");
    validate_native_aliases(prefill_k, prefill_v);
    position_ += seq_len;
}

void QwenVlKvCache::advance(int32_t n_tokens) {
    if (n_tokens <= 0)
        throw std::invalid_argument("Qwen-VL native KV advance must be positive");
    if (n_tokens > max_length_ - position_)
        throw std::runtime_error("Qwen-VL sequence exceeds fixed KV capacity");
    position_ += n_tokens;
}

void QwenVlKvCache::reset() {
    // key_value_lengths hides stale rows, so full-capacity buffers are reusable.
    position_ = 0;
    cache_write_index_ = 0;
    key_value_length_ = 0;
}

std::size_t QwenVlKvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& tensor : cache_k_)
        total += tensor.nbytes();
    for (const auto& tensor : cache_v_)
        total += tensor.nbytes();
    return total;
}

bool QwenVlKvCache::ok() const {
    const auto expected = static_cast<std::size_t>(num_layers_);
    return cache_k_.size() == expected && cache_v_.size() == expected && all_tensors_ok(cache_k_) &&
           all_tensors_ok(cache_v_);
}

} // namespace trtmc
