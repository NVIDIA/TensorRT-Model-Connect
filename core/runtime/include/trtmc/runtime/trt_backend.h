/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// IBackend: the narrow interface implemented by the TensorRT backend DSO.

#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
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

} // namespace trtmc

// C ABI exported by each DSO. The main binary resolves these via dlsym.
extern "C" {
trtmc::IBackend* trtmc_create_backend();
void trtmc_destroy_backend(trtmc::IBackend* backend);
}
