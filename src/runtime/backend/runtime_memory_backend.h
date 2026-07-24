/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

inline constexpr uint32_t kRuntimeMemoryBackendApiVersionV1 = 1;
inline constexpr std::size_t kRuntimeMemoryCudaAlignmentV1 = 256;

// Describes one caller-owned TensorRT I/O allocation. A descriptor is accepted
// only for a tensor declared as deferred when the module was created.
//
// For sequence-shaped storage, valid_tokens is the logical valid extent of
// this tensor: H for a read-only history binding (and may be zero), or Sq for
// an exact current-row output. shape[sequence_axis] describes bound_tokens;
// capacity_bytes must cover the same shape with that dimension expanded to
// capacity_tokens.
struct RuntimeMemoryBindingV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryBindingV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    std::string name;
    void* pointer{nullptr};
    std::size_t capacity_bytes{0};
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};
    std::size_t alignment{kRuntimeMemoryCudaAlignmentV1};
    int32_t device{-1};
    std::shared_ptr<void> lifetime;

    uint64_t valid_tokens{0};
    uint64_t bound_tokens{0};
    uint64_t capacity_tokens{0};
    int32_t sequence_axis{-1};
};

// Shape-only planning descriptor. It lets the memory planner query TensorRT
// activation memory for candidate valid/bound/capacity values before
// allocating the KV slab.
struct RuntimeMemoryShapeV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryShapeV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    std::string name;
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};
    uint64_t valid_tokens{0};
    uint64_t bound_tokens{0};
    uint64_t capacity_tokens{0};
    int32_t sequence_axis{-1};
};

struct RuntimeMemoryAliasShapeV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryAliasShapeV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    RuntimeMemoryShapeV1 input;
    RuntimeMemoryShapeV1 output;
};

// Sets an ordinary, internally allocated dynamic execution input to the
// candidate shape used by context-memory planning. This performs no data copy.
struct RuntimeInputShapeV1 {
    uint32_t struct_size{sizeof(RuntimeInputShapeV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    std::string name;
    std::vector<int64_t> shape;
};

// Declares the graph ABI: output_name must report input_name from
// ICudaEngine::getAliasedInputTensor() after deserialization.
struct RuntimeMemoryAliasPairV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryAliasPairV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    std::string input_name;
    std::string output_name;
};

// Both endpoints are validated before either address becomes visible. The two
// descriptors must describe the same pointer, shape, dtype, capacity and
// sequence bounds.
struct RuntimeMemoryAliasBindingV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryAliasBindingV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    RuntimeMemoryBindingV1 input;
    RuntimeMemoryBindingV1 output;
};

struct RuntimeMemoryModuleOptionsV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryModuleOptionsV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    // API v1 defers only tensors named here (plus alias endpoints). Ordinary
    // TensorRT I/O keeps the legacy internal profile-sized allocation policy.
    std::vector<std::string> deferred_tensor_names;
    std::vector<RuntimeMemoryAliasPairV1> alias_pairs;
};

struct RuntimeMemoryContextRequirementV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryContextRequirementV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    std::size_t capacity_bytes{0};
    std::size_t alignment{kRuntimeMemoryCudaAlignmentV1};
    int32_t device{-1};
};

// A single block may be installed on multiple serially-executed contexts.
// TensorRT forbids concurrent use of the same context-memory block; scheduling
// that serialization remains the pipeline's responsibility.
struct RuntimeMemoryContextBlockV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryContextBlockV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    void* pointer{nullptr};
    std::size_t capacity_bytes{0};
    std::size_t alignment{kRuntimeMemoryCudaAlignmentV1};
    int32_t device{-1};
    std::shared_ptr<void> lifetime;
};

// Read-only accounting reported by one runtime-memory execution context.
// engine_identity is opaque and process-local; callers may use it only to
// deduplicate profile contexts that share one deserialized ICudaEngine.
//
// total_weight_bytes comes from
// ICudaEngine::getEngineStat(kTOTAL_WEIGHTS_SIZE). TensorRT documents that
// value as logical engine weight bytes, not necessarily resident GPU bytes
// when weight streaming is active. The caller must therefore keep resident
// weight accounting unavailable for an actively streamed engine.
struct RuntimeMemoryEngineStatsV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryEngineStatsV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    std::uintptr_t engine_identity{0};
    std::uint64_t total_weight_bytes{0};
    std::uint64_t streamable_weight_bytes{0};
    std::uint64_t weight_streaming_budget_bytes{0};
    std::uint64_t device_output_bytes{0};
    std::uint64_t host_output_staging_bytes{0};
    bool total_weight_bytes_available{false};
    bool weight_streaming_budget_available{false};
    bool cuda_graph_active{false};
};

// Per-module cumulative transfer ledger. The backend updates this at the
// actual cudaMemcpy D2H/D2D call sites and keys counters by tensor name.
// Qualification snapshots before/after each enqueue and subtracts them, so a
// newly introduced cache copy cannot be hidden behind a default zero.
struct RuntimeMemoryTransferCounterV1 {
    std::string tensor_name;
    std::uint64_t device_to_host_bytes{0};
    std::uint64_t device_to_device_bytes{0};
    std::uint64_t device_to_host_events{0};
    std::uint64_t device_to_device_events{0};
    bool runtime_kv_binding{false};
};

struct RuntimeMemoryTransferSnapshotV1 {
    uint32_t struct_size{sizeof(RuntimeMemoryTransferSnapshotV1)};
    uint32_t api_version{kRuntimeMemoryBackendApiVersionV1};
    std::uint64_t event_sequence{0};
    std::vector<RuntimeMemoryTransferCounterV1> counters;
};

struct RuntimeMemoryTransferDeltaV1 {
    std::uint64_t runtime_kv_device_to_host_bytes{0};
    std::uint64_t runtime_kv_device_to_device_bytes{0};
    std::uint64_t runtime_kv_device_to_host_events{0};
    std::uint64_t runtime_kv_device_to_device_events{0};
};

class IRuntimeMemoryTransferLedgerV1 {
  public:
    virtual ~IRuntimeMemoryTransferLedgerV1();
    virtual RuntimeMemoryTransferSnapshotV1 runtime_memory_transfer_snapshot() const = 0;
};

RuntimeMemoryTransferDeltaV1
runtime_memory_transfer_delta(const RuntimeMemoryTransferSnapshotV1& before,
                              const RuntimeMemoryTransferSnapshotV1& after);

// Independent from IRuntimeMemoryModuleV1 so adding accounting does not change
// that binding/control vtable. A mixed backend may omit this capability; the
// receipt then emits explicit nulls instead of inventing zeros.
class IRuntimeMemoryEngineIntrospectionV1 {
  public:
    virtual ~IRuntimeMemoryEngineIntrospectionV1();
    virtual RuntimeMemoryEngineStatsV1 runtime_memory_engine_stats() const noexcept = 0;
};

// Per-module side of the private runtime-memory capability. This intentionally
// does not add virtual slots to the installed ITrtModule interface.
class IRuntimeMemoryModuleV1 {
  public:
    virtual ~IRuntimeMemoryModuleV1();

    virtual void set_runtime_binding_shape(const RuntimeMemoryShapeV1& shape) = 0;
    virtual void set_runtime_alias_pair_shape(const RuntimeMemoryAliasShapeV1& shape) = 0;
    virtual void set_runtime_input_shape(const RuntimeInputShapeV1& shape) = 0;
    virtual void bind_runtime_memory(const RuntimeMemoryBindingV1& binding) = 0;
    virtual void bind_runtime_memory_alias_pair(const RuntimeMemoryAliasBindingV1& binding) = 0;
    virtual RuntimeMemoryContextRequirementV1 context_memory_requirement() = 0;
    virtual void bind_context_memory(const RuntimeMemoryContextBlockV1& block) = 0;
    virtual bool runtime_memory_ready() const noexcept = 0;

    // Selective D2H materialization is needed by split prefill without making
    // it part of the public ITrtModule ABI.
    virtual TensorMap forward_selected(const TensorMap& inputs,
                                       const std::vector<std::string>& host_output_names) = 0;
};

// Standard-TensorRT-only backend capability. Callers discover it with
// dynamic_cast from IBackend; the RTX backend deliberately does not implement
// this interface.
class IRuntimeMemoryBackendV1 {
  public:
    virtual ~IRuntimeMemoryBackendV1();

    virtual uint32_t runtime_memory_api_version() const noexcept {
        return kRuntimeMemoryBackendApiVersionV1;
    }

    virtual std::unique_ptr<ITrtModule>
    create_module_runtime_memory(const void* plan_data, size_t plan_size,
                                 const ModuleCreateOptions& options,
                                 const RuntimeMemoryModuleOptionsV1& memory_options) = 0;

    virtual BackendProfileModules
    create_profile_modules_runtime_memory(const void* plan_data, size_t plan_size,
                                          const ModuleCreateOptions& options,
                                          const std::vector<int32_t>& profile_indices,
                                          const RuntimeMemoryModuleOptionsV1& memory_options) = 0;

    virtual RuntimeMemoryContextRequirementV1
    shared_context_memory_requirement(const std::vector<ITrtModule*>& modules) = 0;

    virtual void bind_shared_context_memory(const std::vector<ITrtModule*>& modules,
                                            const RuntimeMemoryContextBlockV1& block) = 0;
};

} // namespace trtmc
