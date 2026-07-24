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
#include <type_traits>
#include <vector>

namespace trtmc {

// Options for module creation. RTX-specific fields are silently ignored
// by the standard TRT backend.
struct ModuleCreateOptions {
    cudaStream_t stream{nullptr};            // nullptr = backend creates one
    const char* runtime_cache_path{""};      // RTX: JIT kernel cache file path
    bool cuda_graphs{false};                 // RTX: whole-graph CUDA capture
    int32_t optimization_profile{0};         // profile selected for this execution context
    void* distributed_communicator{nullptr}; // TensorRT 11.0+ NCCL communicator, optional
    std::shared_ptr<void> distributed_owner; // keeps communicator alive
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

// Private C++ types cross the core/backend DSO boundary.  A C factory symbol
// alone cannot make those vtables and STL-bearing value types ABI-stable, so
// every backend DSO must first answer this C ABI query.  The loader validates
// the complete contract before calling trtmc_create_backend().
//
// Bump kBackendDsoInterfaceFingerprintV2 whenever an IBackend/ITrtModule
// vtable or a base exchanged type changes.  The runtime-memory header extends
// this contract with its own fingerprint and exact structure sizes.
inline constexpr std::uint32_t kBackendDsoAbiContractVersionV2 = 2;
inline constexpr std::uint64_t kBackendDsoInterfaceFingerprintV2 =
    0x5452544d43414232ULL; // "TRTMCAB2"
inline constexpr std::uint64_t kBackendDsoCapabilityRuntimeMemoryV2 = 1ULL << 0;
inline constexpr std::uint64_t kBackendDsoKnownCapabilitiesV2 =
    kBackendDsoCapabilityRuntimeMemoryV2;
inline constexpr const char kBackendDsoAbiQuerySymbolV2[] = "trtmc_backend_query_abi_contract_v2";

#if defined(__clang__)
inline constexpr std::uint32_t kBackendDsoCompilerId = 2;
inline constexpr std::uint32_t kBackendDsoCompilerVersion =
    __clang_major__ * 1000000U + __clang_minor__ * 1000U + __clang_patchlevel__;
#elif defined(__GNUC__)
inline constexpr std::uint32_t kBackendDsoCompilerId = 1;
inline constexpr std::uint32_t kBackendDsoCompilerVersion =
    __GNUC__ * 1000000U + __GNUC_MINOR__ * 1000U + __GNUC_PATCHLEVEL__;
#else
inline constexpr std::uint32_t kBackendDsoCompilerId = 0;
inline constexpr std::uint32_t kBackendDsoCompilerVersion = 0;
#endif

#if defined(__GXX_ABI_VERSION)
inline constexpr std::uint32_t kBackendDsoCxxAbiVersion = __GXX_ABI_VERSION;
#else
inline constexpr std::uint32_t kBackendDsoCxxAbiVersion = 0;
#endif

#if defined(_GLIBCXX_USE_CXX11_ABI)
inline constexpr std::uint32_t kBackendDsoCxx11StringAbi = _GLIBCXX_USE_CXX11_ABI;
#else
inline constexpr std::uint32_t kBackendDsoCxx11StringAbi = UINT32_MAX;
#endif

#if defined(_LIBCPP_VERSION)
inline constexpr std::uint32_t kBackendDsoStdlibId = 2;
inline constexpr std::uint32_t kBackendDsoStdlibVersion = _LIBCPP_VERSION;
#elif defined(__GLIBCXX__)
inline constexpr std::uint32_t kBackendDsoStdlibId = 1;
inline constexpr std::uint32_t kBackendDsoStdlibVersion = __GLIBCXX__;
#else
inline constexpr std::uint32_t kBackendDsoStdlibId = 0;
inline constexpr std::uint32_t kBackendDsoStdlibVersion = 0;
#endif

struct BackendDsoAbiContractV2 {
    std::uint32_t struct_size{sizeof(BackendDsoAbiContractV2)};
    std::uint32_t contract_version{kBackendDsoAbiContractVersionV2};
    std::uint64_t interface_fingerprint{kBackendDsoInterfaceFingerprintV2};
    std::uint64_t runtime_memory_layout_fingerprint{0};
    std::uint64_t capability_flags{0};

    // Exact compiler/standard-library build contract.  This intentionally
    // fails closed instead of assuming that two separately built C++ DSOs
    // exchange std::string/vector/shared_ptr values compatibly.
    std::uint32_t cxx_standard{static_cast<std::uint32_t>(__cplusplus)};
    std::uint32_t compiler_id{kBackendDsoCompilerId};
    std::uint32_t compiler_version{kBackendDsoCompilerVersion};
    std::uint32_t cxx_abi_version{kBackendDsoCxxAbiVersion};
    std::uint32_t stdlib_id{kBackendDsoStdlibId};
    std::uint32_t stdlib_version{kBackendDsoStdlibVersion};
    std::uint32_t cxx11_string_abi{kBackendDsoCxx11StringAbi};
    std::uint32_t pointer_size{sizeof(void*)};
    std::uint32_t size_t_size{sizeof(std::size_t)};

    // Base backend/core value types and vtable-bearing interfaces.
    std::uint32_t std_string_size{sizeof(std::string)};
    std::uint32_t std_string_alignment{alignof(std::string)};
    std::uint32_t std_vector_size{sizeof(std::vector<void*>)};
    std::uint32_t std_vector_alignment{alignof(std::vector<void*>)};
    std::uint32_t std_shared_ptr_size{sizeof(std::shared_ptr<void>)};
    std::uint32_t std_shared_ptr_alignment{alignof(std::shared_ptr<void>)};
    std::uint32_t std_unique_ptr_size{sizeof(std::unique_ptr<ITrtModule>)};
    std::uint32_t std_unique_ptr_alignment{alignof(std::unique_ptr<ITrtModule>)};
    std::uint32_t dtype_size{sizeof(DType)};
    std::uint32_t tensor_size{sizeof(Tensor)};
    std::uint32_t tensor_alignment{alignof(Tensor)};
    std::uint32_t tensor_map_size{sizeof(TensorMap)};
    std::uint32_t tensor_map_alignment{alignof(TensorMap)};
    std::uint32_t device_tensor_map_size{sizeof(DeviceTensorMap)};
    std::uint32_t device_tensor_map_alignment{alignof(DeviceTensorMap)};
    std::uint32_t tensor_info_size{sizeof(TensorInfo)};
    std::uint32_t tensor_info_alignment{alignof(TensorInfo)};
    std::uint32_t module_create_options_size{sizeof(ModuleCreateOptions)};
    std::uint32_t module_create_options_alignment{alignof(ModuleCreateOptions)};
    std::uint32_t backend_dual_profile_modules_size{sizeof(BackendDualProfileModules)};
    std::uint32_t backend_profile_module_size{sizeof(BackendProfileModule)};
    std::uint32_t backend_profile_modules_size{sizeof(BackendProfileModules)};
    std::uint32_t backend_context_modules_size{sizeof(BackendContextModules)};
    std::uint32_t i_trt_module_size{sizeof(ITrtModule)};
    std::uint32_t i_trt_module_alignment{alignof(ITrtModule)};
    std::uint32_t i_backend_size{sizeof(IBackend)};
    std::uint32_t i_backend_alignment{alignof(IBackend)};

    // Filled by runtime_memory_backend.h even when a backend does not
    // advertise the capability, so an RTX DSO still proves it was built
    // against the same private core declarations.
    std::uint32_t runtime_memory_api_version{0};
    std::uint32_t runtime_memory_binding_size{0};
    std::uint32_t runtime_memory_shape_size{0};
    std::uint32_t runtime_memory_alias_shape_size{0};
    std::uint32_t runtime_input_shape_size{0};
    std::uint32_t runtime_memory_alias_pair_size{0};
    std::uint32_t runtime_memory_alias_binding_size{0};
    std::uint32_t runtime_memory_module_options_size{0};
    std::uint32_t runtime_memory_context_requirement_size{0};
    std::uint32_t runtime_memory_context_block_size{0};
    std::uint32_t runtime_memory_engine_stats_size{0};
    std::uint32_t runtime_memory_transfer_counter_size{0};
    std::uint32_t runtime_memory_transfer_snapshot_size{0};
    std::uint32_t runtime_memory_module_interface_size{0};
    std::uint32_t runtime_memory_backend_interface_size{0};
};

static_assert(std::is_standard_layout_v<BackendDsoAbiContractV2>);
static_assert(std::is_trivially_copyable_v<BackendDsoAbiContractV2>);

inline BackendDsoAbiContractV2 make_backend_dso_abi_contract_v2() noexcept {
    return {};
}

using BackendDsoAbiQueryFnV2 = std::int32_t (*)(BackendDsoAbiContractV2* contract,
                                                std::size_t contract_size) noexcept;

} // namespace trtmc

// C ABI exported by each DSO. The main binary resolves these via dlsym.
extern "C" {
std::int32_t trtmc_backend_query_abi_contract_v2(trtmc::BackendDsoAbiContractV2* contract,
                                                 std::size_t contract_size) noexcept;
trtmc::IBackend* trtmc_create_backend();
void trtmc_destroy_backend(trtmc::IBackend* backend);
}
