/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen_vl/lora_runtime.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace trtmc {
namespace qwen_vl {

namespace {

void validate_static_input(const TensorInfo& info) {
    if (!info.is_input)
        throw std::invalid_argument("Adapter input contract contains output '" + info.name + "'");
    if (info.name.empty())
        throw std::invalid_argument("Adapter input contract contains an empty tensor name");
    if (info.shape.empty() ||
        std::any_of(info.shape.begin(), info.shape.end(), [](int64_t dim) { return dim <= 0; })) {
        throw std::invalid_argument("Adapter input '" + info.name +
                                    "' must have a fixed, non-empty shape");
    }
}

} // namespace

LoraInputBindings::LoraInputBindings(TrtModule& module,
                                           std::vector<TensorInfo> input_contract)
    : module_(module) {
    std::unordered_set<std::string> names;
    names.reserve(input_contract.size());
    entries_.reserve(input_contract.size());
    for (auto& info : input_contract) {
        validate_static_input(info);
        if (!module_.has_input(info.name))
            throw std::invalid_argument("Adapter input '" + info.name +
                                        "' is not present in the execution module");
        if (!names.insert(info.name).second)
            throw std::invalid_argument("Duplicate adapter input '" + info.name + "'");
        auto zero = DeviceTensor::zeros(info.shape, info.dtype, module_.stream());
        entries_.push_back(Entry{std::move(info), std::move(zero)});
        if (!entries_.back().zero.ok())
            throw std::runtime_error("Failed to allocate zero buffer for adapter input '" +
                                     entries_.back().info.name + "'");
    }
    bind_zero_buffers();
}

std::vector<std::string> LoraInputBindings::input_names() const {
    std::vector<std::string> names;
    names.reserve(entries_.size());
    for (const auto& entry : entries_)
        names.push_back(entry.info.name);
    return names;
}

std::vector<TensorInfo> LoraInputBindings::input_info() const {
    std::vector<TensorInfo> infos;
    infos.reserve(entries_.size());
    for (const auto& entry : entries_)
        infos.push_back(entry.info);
    return infos;
}

void LoraInputBindings::bind(const DeviceTensorMap& adapter_tensors) {
    std::unordered_set<std::string> expected;
    expected.reserve(entries_.size());
    for (const auto& entry : entries_)
        expected.insert(entry.info.name);
    for (const auto& [name, tensor] : adapter_tensors) {
        if (expected.find(name) == expected.end())
            throw std::invalid_argument("Unknown adapter tensor '" + name + "'");
        if (tensor == nullptr || !tensor->ok())
            throw std::invalid_argument("Invalid adapter tensor '" + name + "'");
    }
    for (const auto& entry : entries_) {
        const auto it = adapter_tensors.find(entry.info.name);
        if (it == adapter_tensors.end())
            continue;
        const DeviceTensor& tensor = *it->second;
        if (tensor.shape() != entry.info.shape || tensor.dtype() != entry.info.dtype) {
            throw std::invalid_argument("Shape or dtype mismatch for adapter tensor '" +
                                        entry.info.name + "'");
        }
    }

    // Tensor addresses are execution-context state. Do not replace them while
    // a previous enqueue on this module can still read the old addresses.
    module_.sync();
    for (auto& entry : entries_) {
        const auto it = adapter_tensors.find(entry.info.name);
        if (it == adapter_tensors.end()) {
            module_.bind_external(entry.info.name, entry.zero.data());
            continue;
        }
        DeviceTensor& tensor = *it->second;
        module_.bind_external(entry.info.name, tensor.data());
    }
}

void LoraInputBindings::clear() {
    module_.sync();
    bind_zero_buffers();
}

void LoraInputBindings::bind_zero_buffers() {
    for (auto& entry : entries_)
        module_.bind_external(entry.info.name, entry.zero.data());
}

DeviceTensorMap LoraDeviceWeights::device_view() {
    DeviceTensorMap view;
    view.reserve(tensors_.size());
    for (auto& [name, tensor] : tensors_)
        view.emplace(name, &tensor);
    return view;
}

LoraAdapterCache::LoraAdapterCache(std::vector<TensorInfo> input_contract, cudaStream_t upload_stream,
                           std::size_t capacity)
    : upload_stream_(upload_stream), capacity_(capacity) {
    if (capacity_ == 0)
        throw std::invalid_argument("Qwen-VL LoRA cache: capacity must be positive");
    contract_.reserve(input_contract.size());
    for (auto& info : input_contract) {
        validate_static_input(info);
        const std::string name = info.name;
        if (!contract_.emplace(name, std::move(info)).second)
            throw std::invalid_argument("Duplicate adapter input '" + name + "'");
    }
}

void LoraAdapterCache::register_adapter(const std::string& adapter_id, const TensorMap& host_tensors) {
    if (contract_.empty())
        throw std::runtime_error("Qwen-VL LoRA cache: engine has no adapter inputs");
    if (adapter_id.empty())
        throw std::invalid_argument("Qwen-VL LoRA cache: adapter ID must not be empty");
    if (host_tensors.empty())
        throw std::invalid_argument("Qwen-VL LoRA cache: adapter has no tensors");

    auto weights = std::make_shared<LoraDeviceWeights>();
    for (const auto& [name, host] : host_tensors) {
        const auto expected = contract_.find(name);
        if (expected == contract_.end())
            throw std::invalid_argument("Qwen-VL LoRA cache: unknown adapter tensor '" + name + "'");
        if (host.data == nullptr || host.shape != expected->second.shape ||
            host.dtype != expected->second.dtype) {
            throw std::invalid_argument("Qwen-VL LoRA cache: shape or dtype mismatch for tensor '" +
                                        name + "'");
        }
        DeviceTensor device(host.shape, host.dtype, upload_stream_);
        if (!device.ok() || !device.copy_from_host(host.data))
            throw std::runtime_error("Qwen-VL LoRA cache: failed to upload tensor '" + name + "'");
        weights->tensors_.emplace(name, std::move(device));
    }
    if (cudaStreamSynchronize(upload_stream_) != cudaSuccess)
        throw std::runtime_error("Qwen-VL LoRA cache: adapter upload synchronization failed");

    const std::lock_guard<std::mutex> lock(mutex_);
    adapters_[adapter_id] = Entry{std::move(weights), ++clock_};
    evict_if_needed(adapter_id);
}

void LoraAdapterCache::unregister_adapter(const std::string& adapter_id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = adapters_.find(adapter_id);
    if (it == adapters_.end())
        throw std::invalid_argument("Qwen-VL LoRA cache: unknown adapter ID '" + adapter_id + "'");
    // Contexts that already acquired this entry retain shared ownership until
    // their request completes, but new acquisitions fail immediately.
    adapters_.erase(it);
}

std::shared_ptr<LoraDeviceWeights> LoraAdapterCache::acquire(const std::string& adapter_id) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = adapters_.find(adapter_id);
    if (it == adapters_.end())
        throw std::invalid_argument("Qwen-VL LoRA cache: unknown adapter ID '" + adapter_id + "'");
    it->second.last_use = ++clock_;
    return it->second.weights;
}

bool LoraAdapterCache::contains(const std::string& adapter_id) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return adapters_.find(adapter_id) != adapters_.end();
}

std::size_t LoraAdapterCache::size() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return adapters_.size();
}

std::vector<std::string> LoraAdapterCache::adapter_ids() const {
    std::vector<std::string> ids;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        ids.reserve(adapters_.size());
        for (const auto& [id, entry] : adapters_) {
            (void)entry;
            ids.push_back(id);
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

void LoraAdapterCache::evict_if_needed(const std::string& protected_adapter_id) {
    while (adapters_.size() > capacity_) {
        auto victim = adapters_.end();
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (auto it = adapters_.begin(); it != adapters_.end(); ++it) {
            if (it->first == protected_adapter_id || it->second.weights.use_count() > 1)
                continue;
            if (it->second.last_use < oldest) {
                oldest = it->second.last_use;
                victim = it;
            }
        }
        if (victim == adapters_.end()) {
            adapters_.erase(protected_adapter_id);
            throw std::runtime_error(
                "Qwen-VL LoRA cache: capacity is exhausted by adapters in active use");
        }
        adapters_.erase(victim);
    }
}

void LoraBindingContext::select(const std::string& adapter_id) {
    if (adapter_id.empty()) {
        clear();
        return;
    }

    auto weights = cache_.acquire(adapter_id);
    if (weights == active_weights_)
        return;
    auto tensors = weights->device_view();
    bindings_.bind(tensors);
    active_adapter_id_ = adapter_id;
    active_weights_ = std::move(weights);
}

void LoraBindingContext::clear() {
    if (active_adapter_id_.empty())
        return;
    bindings_.clear();
    active_adapter_id_.clear();
    active_weights_.reset();
}

} // namespace qwen_vl
} // namespace trtmc
