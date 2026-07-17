/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// IBackend: virtual interface for TRT backend DSOs.
// Each DSO (libtrtmc_backend_trt.so, libtrtmc_backend_rtx.so) implements
// IBackend and exports C ABI factory functions.

#include "trtmc/runtime/trt_module.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

// Increment whenever the IBackend vtable or any type passed through that
// vtable changes layout. BackendLoader validates this before constructing an
// IBackend, and the factory/destroy symbol names carry the same version so both
// old-core/new-backend and new-core/old-backend installations fail closed.
inline constexpr std::uint32_t kTrtmcBackendApiAbiVersion = 1;

// A caller-owned device allocation that should replace a backend-owned I/O
// allocation while a module is constructed. The allocation must remain valid
// for the module lifetime. Prebinding does not reduce memory consumed while
// deserializing an engine or creating its execution context; both happen before
// the module binds I/O addresses.
struct ModuleExternalBinding {
    std::string tensor_name;
    void* device_ptr{nullptr};
    std::size_t capacity_bytes{0};
};

// Options for module creation. RTX-specific fields are silently ignored
// by the standard TRT backend.
struct ModuleCreateOptions {
    cudaStream_t stream{nullptr};            // nullptr = backend creates one
    const char* runtime_cache_path{""};      // RTX: JIT kernel cache file path
    bool cuda_graphs{false};                 // RTX: whole-graph CUDA capture
    int32_t optimization_profile{0};         // profile selected for this execution context
    void* distributed_communicator{nullptr}; // TensorRT 11.0+ NCCL communicator, optional
    std::shared_ptr<void> distributed_owner; // keeps communicator alive
    // Static engine I/O to bind before backend-owned I/O buffers or host output
    // staging are allocated. Empty preserves the historical behavior. A single
    // binding set cannot be shared by multiple simultaneously-live profiles.
    std::vector<ModuleExternalBinding> external_bindings;
};

// Two modules created from a single engine, one per optimization profile.
// Both share the engine (weights live once on GPU) and the same CUDA stream.
// When the engine has fewer than two profiles, `prefill` is null and `decode`
// holds the only context.
struct BackendDualProfileModules {
    std::unique_ptr<ITrtModule> prefill; // profile 0 — batched-Sq prefill (null if single-profile)
    std::unique_ptr<ITrtModule> decode;  // profile 1, or the only profile if single-profile
};

struct BackendProfileModule {
    int32_t profile_idx{0};
    std::unique_ptr<ITrtModule> module;
};

struct BackendProfileModules {
    std::vector<BackendProfileModule> modules;
};

// One execution module per request lane. All modules share a single
// deserialized engine, while each ModuleCreateOptions entry may provide an
// independent CUDA stream and distributed-runtime ownership.
struct BackendContextModules {
    std::vector<std::unique_ptr<ITrtModule>> modules;
};

// Per-DSO backend. Holds shared state (TRT runtime, RTX runtime cache).
// One IBackend creates all ITrtModule instances for a pipeline.
class IBackend {
  public:
    virtual ~IBackend() = default;

    // Deserialize an engine plan and create a module.
    virtual std::unique_ptr<ITrtModule> create_module(const void* plan_data, size_t plan_size,
                                                      const ModuleCreateOptions& options) = 0;

    // Deserialize once, create two execution contexts sharing the engine —
    // profile 0 for prefill, profile 1 for decode. Falls back to single-
    // profile (prefill=null) when the engine only has one profile.
    virtual BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) = 0;

    // Deserialize once and create one execution context per requested profile.
    // Returned modules share engine weights and stream ownership.
    virtual BackendProfileModules
    create_profile_modules(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<int32_t>& profile_indices) = 0;

    // Deserialize once and create one execution context for each options
    // entry, using its requested optimization profile. Intended for
    // concurrent request lanes.
    virtual BackendContextModules
    create_context_modules(const void* plan_data, size_t plan_size,
                           const std::vector<ModuleCreateOptions>& options) = 0;

    // Backend identity: "trt" or "trt_rtx"
    virtual const char* name() const = 0;
};

} // namespace trtmc

// C ABI exported by each DSO. The main binary resolves these via dlsym.
extern "C" {
std::uint32_t trtmc_backend_api_abi_version();
trtmc::IBackend* trtmc_create_backend_v1();
void trtmc_destroy_backend_v1(trtmc::IBackend* backend);
}
