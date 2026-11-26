/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/laguna/runtime/kv_cache.h"

#include "families/laguna/runtime/tensor_names.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace trtmc::laguna {

namespace {

// Additive score of a hidden key (stale cache row or future token).
constexpr float kMaskedScore = -1.0e4F;

void require_shape(const ITrtModule& module, const std::string& name,
                   const std::vector<std::int64_t>& shape) {
    if (module.tensor_shape(name) != shape)
        throw std::runtime_error("laguna engine tensor " + name + " does not match the KV cache");
}

} // namespace

KvCache::KvCache(std::int32_t num_layers, std::int32_t max_length, std::int32_t kv_width,
                 cudaStream_t stream, DType dtype)
    : num_layers_(num_layers), max_length_(max_length), kv_width_(kv_width), stream_(stream),
      row_bytes_(static_cast<std::size_t>(kv_width) * dtype_size(dtype)) {
    for (std::int32_t layer = 0; layer < num_layers; ++layer) {
        cache_k_.emplace_back(std::vector<std::int64_t>{max_length, kv_width}, dtype, stream);
        cache_v_.emplace_back(std::vector<std::int64_t>{max_length, kv_width}, dtype, stream);
        present_k_.emplace_back(std::vector<std::int64_t>{1, kv_width}, dtype, stream);
        present_v_.emplace_back(std::vector<std::int64_t>{1, kv_width}, dtype, stream);
    }
    // Masked keys get zero attention weight, but a NaN or Inf bit pattern left
    // in a never-written row would still poison the weighted value sum.
    for (auto* tensors : {&cache_k_, &cache_v_, &present_k_, &present_v_}) {
        for (auto& tensor : *tensors) {
            if (tensor.ok() &&
                cudaMemsetAsync(tensor.data(), 0, tensor.nbytes(), stream) != cudaSuccess)
                throw std::runtime_error("laguna KV cache initialisation failed");
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess)
        throw std::runtime_error("laguna KV cache initialisation failed");
}

bool KvCache::ok() const {
    const auto all_ok = [](const std::vector<DeviceTensor>& tensors) {
        return std::all_of(tensors.begin(), tensors.end(),
                           [](const DeviceTensor& tensor) { return tensor.ok(); });
    };
    return static_cast<std::int32_t>(cache_k_.size()) == num_layers_ && all_ok(cache_k_) &&
           all_ok(cache_v_) && all_ok(present_k_) && all_ok(present_v_);
}

void KvCache::bind_prefill(ITrtModule& module) {
    const std::vector<std::int64_t> cache_shape{max_length_, kv_width_};
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        require_shape(module, layer_tensor_name("cache_k", layer), cache_shape);
        require_shape(module, layer_tensor_name("cache_v", layer), cache_shape);
        module.bind_external(layer_tensor_name("cache_k", layer), cache_k_[index].data());
        module.bind_external(layer_tensor_name("cache_v", layer), cache_v_[index].data());
    }
}

void KvCache::bind_decode(ITrtModule& module) {
    bind_prefill(module);
    if (module.tensor_shape("attention_mask") != std::vector<std::int64_t>{1, max_length_ + 1})
        throw std::runtime_error("laguna decode context has an invalid attention-mask contract");
    const std::vector<std::int64_t> present_shape{1, kv_width_};
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        require_shape(module, layer_tensor_name("present_k", layer), present_shape);
        require_shape(module, layer_tensor_name("present_v", layer), present_shape);
        module.bind_external(layer_tensor_name("present_k", layer), present_k_[index].data());
        module.bind_external(layer_tensor_name("present_v", layer), present_v_[index].data());
    }
}

void KvCache::prepare(TensorMap& inputs, std::int32_t rows) {
    if (rows <= 0 || position_ + rows > max_length_)
        throw std::runtime_error("laguna sequence exceeds the KV cache capacity");
    positions_.resize(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row)
        positions_[static_cast<std::size_t>(row)] = position_ + row;
    Tensor position;
    position.data = positions_.data();
    position.shape = {rows};
    position.dtype = DType::kInt32;
    inputs["position_id"] = position;

    // Columns [0, max_length) are cache rows (valid up to position_); columns
    // max_length + j are the new rows, causal among themselves.
    const std::int32_t width = max_length_ + rows;
    mask_.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(width), kMaskedScore);
    for (std::int32_t row = 0; row < rows; ++row) {
        auto* line = mask_.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
        std::fill(line, line + position_, 0.0F);
        std::fill(line + max_length_, line + max_length_ + row + 1, 0.0F);
    }
    Tensor mask;
    mask.data = mask_.data();
    mask.shape = {rows, width};
    mask.dtype = DType::kFloat32;
    inputs["attention_mask"] = mask;
}

void KvCache::append(const std::vector<const void*>& present_k,
                     const std::vector<const void*>& present_v, std::int32_t rows) {
    if (rows <= 0 || position_ + rows > max_length_)
        throw std::runtime_error("laguna KV append exceeds the cache capacity");
    if (static_cast<std::int32_t>(present_k.size()) != num_layers_ ||
        static_cast<std::int32_t>(present_v.size()) != num_layers_)
        throw std::runtime_error("laguna KV append needs one K and one V buffer per layer");
    const std::size_t offset = static_cast<std::size_t>(position_) * row_bytes_;
    const std::size_t bytes = static_cast<std::size_t>(rows) * row_bytes_;
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        auto* k = static_cast<std::uint8_t*>(cache_k_[index].data()) + offset;
        auto* v = static_cast<std::uint8_t*>(cache_v_[index].data()) + offset;
        if (cudaMemcpyAsync(k, present_k[index], bytes, cudaMemcpyDeviceToDevice, stream_) !=
                cudaSuccess ||
            cudaMemcpyAsync(v, present_v[index], bytes, cudaMemcpyDeviceToDevice, stream_) !=
                cudaSuccess)
            throw std::runtime_error("laguna KV append copy failed");
    }
    position_ += rows;
}

void KvCache::append_decoded() {
    std::vector<const void*> k;
    std::vector<const void*> v;
    for (std::int32_t layer = 0; layer < num_layers_; ++layer) {
        k.push_back(present_k_[static_cast<std::size_t>(layer)].data());
        v.push_back(present_v_[static_cast<std::size_t>(layer)].data());
    }
    append(k, v, 1);
}

} // namespace trtmc::laguna
