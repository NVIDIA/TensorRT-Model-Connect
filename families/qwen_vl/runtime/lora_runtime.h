/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc {
namespace qwen_vl {

// Context-local bindings for the fixed-shape LoRA inputs emitted by the
// Qwen-VL builder.
class LoraInputBindings {
  public:
    LoraInputBindings(ITrtModule& module, std::vector<TensorInfo> input_contract);

    bool enabled() const { return !entries_.empty(); }
    cudaStream_t stream() const { return module_.stream(); }
    std::vector<std::string> input_names() const;
    std::vector<TensorInfo> input_info() const;

    // Missing contract inputs are bound to zero, allowing an artifact to
    // target a subset of the adapter inputs compiled into the engine.
    void bind(const DeviceTensorMap& adapter_tensors);
    void clear();

  private:
    struct Entry {
        TensorInfo info;
        DeviceTensor zero;
    };

    ITrtModule& module_;
    std::vector<Entry> entries_;

    void bind_zero_buffers();
};

// Immutable device tensors for one normalized adapter. Shared ownership pins
// the buffers while one or more execution contexts have them bound.
class LoraDeviceWeights {
  public:
    DeviceTensorMap device_view();

  private:
    friend class LoraAdapterCache;
    std::unordered_map<std::string, DeviceTensor> tensors_;
};

// GPU-resident adapter registry for one Qwen-VL engine tensor contract. Cache
// entries can be shared by multiple context-local LoraBindingContext objects.
class LoraAdapterCache {
  public:
    LoraAdapterCache(std::vector<TensorInfo> input_contract, cudaStream_t upload_stream,
                     std::size_t capacity = 4);

    void register_adapter(const std::string& adapter_id, const TensorMap& host_tensors);
    void unregister_adapter(const std::string& adapter_id);
    std::shared_ptr<LoraDeviceWeights> acquire(const std::string& adapter_id);

    bool contains(const std::string& adapter_id) const;
    std::size_t size() const;
    std::vector<std::string> adapter_ids() const;

  private:
    struct Entry {
        std::shared_ptr<LoraDeviceWeights> weights;
        uint64_t last_use{0};
    };

    std::unordered_map<std::string, TensorInfo> contract_;
    cudaStream_t upload_stream_{nullptr};
    std::size_t capacity_{4};
    uint64_t clock_{0};
    std::unordered_map<std::string, Entry> adapters_;
    mutable std::mutex mutex_;

    void evict_if_needed(const std::string& protected_adapter_id);
};

// One execution context's active adapter selection. Holding active_weights_
// pins the cache entry until this context is cleared or selects another one.
class LoraBindingContext {
  public:
    LoraBindingContext(LoraInputBindings& bindings, LoraAdapterCache& cache)
        : bindings_(bindings), cache_(cache) {}

    void select(const std::string& adapter_id);
    void clear();

    const std::string& active_adapter_id() const { return active_adapter_id_; }

  private:
    LoraInputBindings& bindings_;
    LoraAdapterCache& cache_;
    std::string active_adapter_id_;
    std::shared_ptr<LoraDeviceWeights> active_weights_;
};

} // namespace qwen_vl
} // namespace trtmc
