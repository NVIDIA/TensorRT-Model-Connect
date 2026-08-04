/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/deepseek_ocr/kv_cache.h"

#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

void validate_scalar_input(TrtModule& module, const std::string& name) {
    if (!module.has_input(name) || module.tensor_dtype(name) != DType::kInt32 ||
        module.tensor_shape(name) != std::vector<int64_t>{1}) {
        throw std::runtime_error("DeepSeek-OCR native KV input '" + name + "' must be int32 [1]");
    }
}

bool valid_cache_shape(const std::vector<int64_t>& shape, int32_t max_length, int32_t kv_dim) {
    return shape.size() == 4 && shape[0] == 1 && shape[2] == static_cast<int64_t>(max_length) &&
           shape[1] > 0 && shape[3] > 0 && shape[1] * shape[3] == kv_dim;
}

void validate_cache_pair(TrtModule& module, const std::string& cache_name,
                         const std::string& present_name, int32_t max_length, int32_t kv_dim,
                         DType cache_dtype) {
    if (!module.has_input(cache_name) || !module.has_output(present_name)) {
        throw std::runtime_error("DeepSeek-OCR native KV engine is missing cache/present pair '" +
                                 cache_name + "'/'" + present_name + "'");
    }
    const auto cache_shape = module.tensor_shape(cache_name);
    if (!valid_cache_shape(cache_shape, max_length, kv_dim) ||
        module.tensor_shape(present_name) != cache_shape) {
        throw std::runtime_error("DeepSeek-OCR native KV cache/present tensors must share static "
                                 "[1,Hkv,max_length,D] shape");
    }
    if (module.tensor_dtype(cache_name) != cache_dtype ||
        module.tensor_dtype(present_name) != cache_dtype) {
        throw std::runtime_error(
            "DeepSeek-OCR native KV cache dtype must match BF16 model precision");
    }
}

} // namespace

DeepseekOcrKvCache::DeepseekOcrKvCache(int32_t num_layers, int32_t max_length, int32_t kv_dim,
                                       cudaStream_t stream, DType cache_dtype,
                                       DeepseekOcrKvCacheNames names)
    : num_layers_(num_layers), max_length_(max_length), kv_dim_(kv_dim), cache_dtype_(cache_dtype),
      names_(std::move(names)) {
    if (num_layers_ <= 0 || max_length_ <= 0 || kv_dim_ <= 0)
        throw std::invalid_argument("DeepseekOcrKvCache dimensions must be positive");
    if (cache_dtype_ != DType::kBFloat16)
        throw std::invalid_argument("DeepseekOcrKvCache requires BF16 storage");

    if (names_.cache_k.empty()) {
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            const auto suffix = "_" + std::to_string(layer);
            names_.cache_k.push_back("cache_k" + suffix);
            names_.cache_v.push_back("cache_v" + suffix);
            names_.present_k.push_back("present_k" + suffix);
            names_.present_v.push_back("present_v" + suffix);
        }
    }
    const auto count = static_cast<std::size_t>(num_layers_);
    if (names_.cache_k.size() != count || names_.cache_v.size() != count ||
        names_.present_k.size() != count || names_.present_v.size() != count) {
        throw std::invalid_argument("DeepseekOcrKvCache per-layer tensor name count mismatch");
    }

    cache_k_.reserve(count);
    cache_v_.reserve(count);
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        cache_k_.emplace_back(std::vector<int64_t>{max_length_, kv_dim_}, cache_dtype_, stream);
        if (!cache_k_.back().ok())
            return;
        cache_v_.emplace_back(std::vector<int64_t>{max_length_, kv_dim_}, cache_dtype_, stream);
        if (!cache_v_.back().ok())
            return;
    }
    reset();
}

void DeepseekOcrKvCache::validate_and_bind(TrtModule& module) {
    validate_scalar_input(module, names_.cache_write_indices);
    validate_scalar_input(module, names_.key_value_lengths);
    for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        validate_cache_pair(module, names_.cache_k[index], names_.present_k[index], max_length_,
                            kv_dim_, cache_dtype_);
        validate_cache_pair(module, names_.cache_v[index], names_.present_v[index], max_length_,
                            kv_dim_, cache_dtype_);

        module.bind_external(names_.cache_k[index], cache_k_[index].data());
        module.bind_external(names_.cache_v[index], cache_v_[index].data());
        if (module.device_ptr(names_.cache_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.present_k[index]) != cache_k_[index].data() ||
            module.device_ptr(names_.cache_v[index]) != cache_v_[index].data() ||
            module.device_ptr(names_.present_v[index]) != cache_v_[index].data()) {
            throw std::runtime_error(
                "DeepSeek-OCR native KV cache/present aliasing was not preserved");
        }
    }
}

void DeepseekOcrKvCache::bind_to(TrtModule& module) {
    if (!module.has_input(names_.position_id))
        throw std::runtime_error("DeepSeek-OCR native KV engine is missing position_id");
    validate_and_bind(module);
}

void DeepseekOcrKvCache::write_position_input(TensorMap& inputs, int32_t seq_len) {
    position_ids_.resize(static_cast<std::size_t>(seq_len));
    for (int32_t offset = 0; offset < seq_len; ++offset)
        position_ids_[static_cast<std::size_t>(offset)] = position_ + offset;
    inputs[names_.position_id] = Tensor{position_ids_.data(), {seq_len}, DType::kInt32};
}

void DeepseekOcrKvCache::prepare_step(TensorMap& inputs, int32_t seq_len) {
    if (seq_len <= 0)
        throw std::invalid_argument("DeepSeek-OCR native KV step length must be positive");
    if (seq_len > max_length_ - position_)
        throw std::runtime_error("DeepSeek-OCR sequence exceeds the fixed KV cache capacity");
    write_position_input(inputs, seq_len);
    cache_write_index_ = position_;
    key_value_length_ = position_ + seq_len;
    inputs[names_.cache_write_indices] = Tensor{&cache_write_index_, {1}, DType::kInt32};
    inputs[names_.key_value_lengths] = Tensor{&key_value_length_, {1}, DType::kInt32};
}

void DeepseekOcrKvCache::advance(int32_t n_tokens) {
    if (n_tokens <= 0)
        throw std::invalid_argument("DeepSeek-OCR native KV advance must be positive");
    if (n_tokens > max_length_ - position_)
        throw std::runtime_error("DeepSeek-OCR sequence exceeds the fixed KV cache capacity");
    position_ += n_tokens;
}

void DeepseekOcrKvCache::reset() {
    position_ = 0;
    cache_write_index_ = 0;
    key_value_length_ = 0;
}

std::size_t DeepseekOcrKvCache::device_memory_bytes() const {
    std::size_t total = 0;
    for (const auto& tensor : cache_k_)
        total += tensor.nbytes();
    for (const auto& tensor : cache_v_)
        total += tensor.nbytes();
    return total;
}

bool DeepseekOcrKvCache::ok() const {
    if (cache_k_.size() != static_cast<std::size_t>(num_layers_) ||
        cache_v_.size() != static_cast<std::size_t>(num_layers_))
        return false;
    return std::all_of(cache_k_.begin(), cache_k_.end(),
                       [](const DeviceTensor& tensor) { return tensor.ok(); }) &&
           std::all_of(cache_v_.begin(), cache_v_.end(),
                       [](const DeviceTensor& tensor) { return tensor.ok(); });
}

} // namespace trtmc
