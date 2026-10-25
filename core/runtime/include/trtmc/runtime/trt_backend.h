/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// IBackend: the narrow interface implemented by the TensorRT backend DSO.

#include "trtmc/bundle.h"
#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

struct ModuleCreateOptions {
    cudaStream_t stream{nullptr};            // nullptr = backend creates one
    void* distributed_communicator{nullptr}; // TensorRT 11.0+ NCCL communicator, optional
    std::shared_ptr<void> distributed_owner; // keeps communicator alive
    const char* runtime_cache_path{""};      // TensorRT-RTX JIT cache, optional
    bool cuda_graphs{false};                 // TensorRT-RTX whole-graph capture
};

struct ModuleExternalBinding {
    std::string tensor_name;
    void* device_ptr{nullptr};
    std::size_t capacity_bytes{0};
};

// Two modules created from an engine that has at least two optimization
// profiles. Both share the engine and CUDA stream.
struct BackendDualProfileModules {
    std::unique_ptr<ITrtModule> prefill; // profile 0
    std::unique_ptr<ITrtModule> decode;  // profile 1
};

// Per-DSO backend. Holds the TensorRT runtime state used by its modules.
// One IBackend creates all ITrtModule instances for a pipeline.
class IBackend {
  public:
    virtual ~IBackend() = default;

    // Deserialize an engine plan and create a module.
    virtual std::unique_ptr<ITrtModule> create_module(const void* plan_data, size_t plan_size,
                                                      const ModuleCreateOptions& options) = 0;

    virtual std::unique_ptr<ITrtModule>
    create_module_prebound(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<ModuleExternalBinding>& external_bindings) = 0;

    // Deserialize once and create contexts for profiles 0 and 1. Engines with
    // fewer than two profiles are rejected.
    virtual BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) = 0;

    // Backend identity written into the bundle header (currently "trt").
    virtual const char* name() const = 0;
};

// Optional capability, separate from IBackend to preserve its existing vtable.
// Ownership transfers to the backend, which may retain the reader with its engine.
class IStreamBackend {
  public:
    virtual ~IStreamBackend() = default;
    virtual bool supports_plan_streaming() const noexcept = 0;
    virtual std::unique_ptr<ITrtModule>
    create_module_from_stream(std::unique_ptr<IPlanReader> plan, const ModuleCreateOptions& options,
                              const std::vector<ModuleExternalBinding>& bindings) = 0;
};

// Families choose when to load an engine. This helper only selects a supported
// input mechanism. A failed streamed load is never retried through buffering.
inline std::unique_ptr<ITrtModule>
create_module_from_reader(IBackend& backend, std::unique_ptr<IPlanReader> plan,
                          const ModuleCreateOptions& options,
                          const std::vector<ModuleExternalBinding>& bindings = {}) {
    if (!plan)
        throw std::invalid_argument("Serialized plan reader must not be null");
    if (auto* streamed = dynamic_cast<IStreamBackend*>(&backend);
        streamed != nullptr && streamed->supports_plan_streaming())
        return streamed->create_module_from_stream(std::move(plan), options, bindings);

    if (plan->size() > std::numeric_limits<std::size_t>::max() ||
        !plan->seek(0, PlanSeekOrigin::begin))
        throw std::runtime_error("Cannot buffer serialized plan source");
    std::vector<char> bytes(static_cast<std::size_t>(plan->size()));
    std::size_t done = 0;
    while (done < bytes.size()) {
        const auto count = plan->read(bytes.data() + done, bytes.size() - done);
        if (count == 0 || count > bytes.size() - done)
            throw std::runtime_error("Incomplete serialized plan source");
        done += count;
    }
    return bindings.empty()
               ? backend.create_module(bytes.data(), bytes.size(), options)
               : backend.create_module_prebound(bytes.data(), bytes.size(), options, bindings);
}

} // namespace trtmc

// C ABI exported by each DSO. The main binary resolves these via dlsym.
extern "C" {
trtmc::IBackend* trtmc_create_backend();
void trtmc_destroy_backend(trtmc::IBackend* backend);
}
