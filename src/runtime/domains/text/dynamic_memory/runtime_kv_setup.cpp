/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_kv_setup.h"

#include "runtime/backend/runtime_memory_backend.h"
#include "trtmc/runtime/trt_module.h"

#include <algorithm>
#include <cctype>
#include <cuda_runtime_api.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace trtmc {
namespace {

thread_local RuntimeDeviceMemoryQualificationObserver runtime_device_memory_qualification_observer;
thread_local RuntimeDeviceMemoryQualificationPreSnapshotAction
    runtime_device_memory_qualification_pre_snapshot_action;

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error(std::string(what) + " overflows uint64");
    return lhs * rhs;
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
        throw std::overflow_error(std::string(what) + " overflows uint64");
    return lhs + rhs;
}

std::uint64_t recompute_kv_bytes_per_token(const RuntimeKvGraphLayout& layout) {
    const auto width = checked_mul(layout.kv_head_count, layout.head_dim, "Runtime KV row width");
    const auto row_bytes = checked_mul(width, dtype_size(layout.dtype), "Runtime KV row bytes");
    const auto layer_pair = checked_mul(row_bytes, std::uint64_t{2}, "Runtime KV K/V layer bytes");
    return checked_mul(layer_pair, layout.layer_count, "Runtime KV bytes per token");
}

std::uint64_t staging_capacity_tokens(const RuntimeKvGraphLayout& layout, std::uint64_t capacity) {
    return std::min<std::uint64_t>(layout.prefill_chunk_limit, capacity);
}

std::uint64_t staging_bytes(const RuntimeKvSetupRequest& request, std::uint64_t capacity) {
    return checked_mul(staging_capacity_tokens(request.layout, capacity),
                       request.expected_kv_bytes_per_token, "Runtime KV current-row staging bytes");
}

bool is_lower_sha256(const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char character) {
               return std::isdigit(character) != 0 ||
                      (character >= static_cast<unsigned char>('a') &&
                       character <= static_cast<unsigned char>('f'));
           });
}

std::uint64_t profile_sequence_limit(const ITrtModule& module, const std::string& tensor_name) {
    const auto rank = module.input_rank(tensor_name);
    const int32_t sequence_axis = rank == 4 ? 2 : rank == 2 || rank == 1 ? 0 : -1;
    if (sequence_axis < 0) {
        throw std::invalid_argument("Runtime KV tensor '" + tensor_name +
                                    "' has an unsupported profile rank");
    }
    const auto shape =
        module.input_profile_shape(tensor_name, module.profile_idx(), ProfileShapeSelector::kMax);
    if (static_cast<std::size_t>(sequence_axis) >= shape.size() ||
        shape[static_cast<std::size_t>(sequence_axis)] <= 0) {
        throw std::invalid_argument("Runtime KV tensor '" + tensor_name +
                                    "' has no positive profile MAX");
    }
    return static_cast<std::uint64_t>(shape[static_cast<std::size_t>(sequence_axis)]);
}

void validate_request(const RuntimeKvSetupRequest& request) {
    if (request.layout.layer_count == 0 || request.layout.kv_head_count == 0 ||
        request.layout.head_dim == 0 || request.layout.capacity_tokens == 0 ||
        request.layout.prefill_chunk_limit == 0) {
        throw std::invalid_argument("Runtime KV setup has an invalid graph layout");
    }
    if (request.layout.prefill_chunk_limit > request.layout.capacity_tokens)
        throw std::invalid_argument("Runtime KV prefill chunk exceeds model context limit");
    if (request.roles.empty())
        throw std::invalid_argument("Runtime KV setup requires execution roles");
    std::size_t prefill_count = 0;
    std::vector<std::uint64_t> decode_limits;
    std::set<ITrtModule*> modules;
    for (const auto& role : request.roles) {
        if (role.module == nullptr)
            throw std::invalid_argument("Runtime KV setup has a null execution role");
        if (!modules.insert(role.module).second) {
            throw std::invalid_argument("Runtime KV setup requires one execution context per role");
        }
        if (dynamic_cast<IRuntimeMemoryModuleV1*>(role.module) == nullptr) {
            throw std::invalid_argument("Runtime KV setup requires USER_MANAGED TensorRT modules");
        }
        if (role.profile_limit == 0 || role.profile_limit > request.layout.capacity_tokens) {
            throw std::invalid_argument("Runtime KV execution role has an invalid profile limit");
        }
        if (profile_sequence_limit(*role.module, request.layout.names.token_id) !=
            (role.kind == RuntimeKvExecutionRoleKind::kPrefill
                 ? static_cast<std::uint64_t>(request.layout.prefill_chunk_limit)
                 : std::uint64_t{1})) {
            throw std::runtime_error(
                "Runtime KV execution role has an unexpected token profile MAX");
        }
        for (const auto& name : request.layout.names.cache_k) {
            if (profile_sequence_limit(*role.module, name) != role.profile_limit) {
                throw std::runtime_error(
                    "Runtime KV execution role cache profile disagrees with its limit");
            }
        }
        for (const auto& name : request.layout.names.cache_v) {
            if (profile_sequence_limit(*role.module, name) != role.profile_limit) {
                throw std::runtime_error(
                    "Runtime KV execution role cache profile disagrees with its limit");
            }
        }
        if (role.kind == RuntimeKvExecutionRoleKind::kPrefill) {
            ++prefill_count;
            if (role.profile_limit != request.layout.capacity_tokens) {
                throw std::runtime_error(
                    "Runtime KV prefill profile does not cover model context limit");
            }
        } else {
            decode_limits.push_back(role.profile_limit);
        }
    }
    if (prefill_count != 1 || decode_limits.empty()) {
        throw std::invalid_argument(
            "Runtime KV setup requires exactly one prefill and at least one decode role");
    }
    if (request.expected_active_kv_profile_limits.empty()) {
        throw std::invalid_argument(
            "Runtime KV setup requires bundle history-bound profile limits");
    }
    for (std::size_t i = 0; i < request.expected_active_kv_profile_limits.size(); ++i) {
        const auto limit = request.expected_active_kv_profile_limits[i];
        if (limit == 0 || limit > request.layout.capacity_tokens ||
            (i > 0 && limit <= request.expected_active_kv_profile_limits[i - 1])) {
            throw std::invalid_argument(
                "Bundle history-bound profile limits must be positive and strictly increasing");
        }
    }
    if (request.expected_active_kv_profile_limits.back() != request.layout.capacity_tokens) {
        throw std::invalid_argument(
            "Bundle history-bound profile limits do not reach model context limit");
    }
    if (request.module_residency_reserve_bytes_by_profile.size() !=
        request.expected_active_kv_profile_limits.size()) {
        throw std::invalid_argument(
            "Bundle module-residency reserve table does not align with profile limits");
    }
    if (request.module_residency_reserve_bytes_by_profile.empty() ||
        request.module_residency_reserve_bytes_by_profile.front() == 0 ||
        !std::is_sorted(request.module_residency_reserve_bytes_by_profile.begin(),
                        request.module_residency_reserve_bytes_by_profile.end())) {
        throw std::invalid_argument(
            "Bundle module-residency reserve table must be positive and nondecreasing");
    }
    if (!is_lower_sha256(request.module_residency_plan_set_sha256) ||
        !is_lower_sha256(request.module_residency_evidence_sha256) ||
        (request.module_residency_cuda_module_loading_mode != "lazy" &&
         request.module_residency_cuda_module_loading_mode != "eager")) {
        throw std::invalid_argument(
            "Bundle module-residency calibration provenance is incomplete");
    }
    std::sort(decode_limits.begin(), decode_limits.end());
    if (std::adjacent_find(decode_limits.begin(), decode_limits.end()) != decode_limits.end()) {
        throw std::runtime_error("Runtime KV engine has duplicate decode profile limits");
    }
    if (decode_limits != request.expected_active_kv_profile_limits) {
        throw std::runtime_error(
            "Runtime KV engine decode profiles do not match the bundle contract");
    }
    if (!request.allocator)
        throw std::invalid_argument("Runtime KV setup requires a device allocator");

    const auto recomputed = recompute_kv_bytes_per_token(request.layout);
    if (request.expected_kv_bytes_per_token == 0 ||
        request.expected_kv_bytes_per_token != recomputed) {
        throw std::invalid_argument(
            "Bundle KV bytes-per-token does not match the engine graph layout");
    }
}

RuntimeDeviceMemorySnapshot cuda_memory_snapshot_after_sync(const char* phase) {
    int current_device = 0;
    if (cudaGetDevice(&current_device) != cudaSuccess)
        throw std::runtime_error("Unable to query CUDA device for runtime KV setup");
    if (current_device < 0)
        throw std::runtime_error("CUDA returned an invalid device for runtime KV setup");
    const auto sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to synchronize ") + phase + ": " +
                                 cudaGetErrorString(sync_status));
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to query GPU memory ") + phase + ": " +
                                 cudaGetErrorString(status));
    }
    return RuntimeDeviceMemorySnapshot{static_cast<std::uint32_t>(current_device),
                                       static_cast<std::uint64_t>(free_bytes),
                                       static_cast<std::uint64_t>(total_bytes)};
}

struct RuntimeReceiptEngineAccounting {
    std::uint64_t engine_weight_bytes{0};
    std::uint64_t resident_weight_bytes{0};
    std::uint32_t resident_weight_copy_count{0};
    std::uint64_t ordinary_device_input_bytes{0};
    std::uint64_t ordinary_device_output_bytes{0};
    std::uint64_t host_output_staging_bytes{0};
    bool engine_weight_bytes_available{false};
    bool resident_weight_bytes_available{false};
    bool resident_weight_copy_count_available{false};
    bool module_allocation_bytes_available{false};
    bool weight_streaming_active{false};
    bool cuda_graph_private_bytes_available{false};
};

RuntimeReceiptEngineAccounting
query_engine_accounting(const std::vector<RuntimeKvExecutionRole>& roles) {
    RuntimeReceiptEngineAccounting accounting;
    std::unordered_map<std::uintptr_t, RuntimeMemoryEngineStatsV1> engines;
    bool all_introspection_available = true;
    bool all_engine_weights_available = true;
    bool all_engine_identities_available = true;
    bool all_resident_weights_available = true;
    bool any_cuda_graph_active = false;
    std::unordered_set<const ITrtModule*> accounted_modules;

    for (const auto& role : roles) {
        if (!accounted_modules.insert(role.module).second)
            continue;
        const auto* introspection =
            dynamic_cast<const IRuntimeMemoryEngineIntrospectionV1*>(role.module);
        if (introspection == nullptr) {
            all_introspection_available = false;
            all_engine_weights_available = false;
            all_engine_identities_available = false;
            all_resident_weights_available = false;
            continue;
        }
        const auto stats = introspection->runtime_memory_engine_stats();
        if (stats.struct_size < sizeof(RuntimeMemoryEngineStatsV1) ||
            stats.api_version != kRuntimeMemoryBackendApiVersionV1) {
            throw std::runtime_error("Runtime-memory engine accounting has an incompatible ABI");
        }
        accounting.ordinary_device_input_bytes =
            checked_add(accounting.ordinary_device_input_bytes, stats.ordinary_device_input_bytes,
                        "Runtime ordinary device input accounting");
        accounting.ordinary_device_output_bytes =
            checked_add(accounting.ordinary_device_output_bytes, stats.ordinary_device_output_bytes,
                        "Runtime ordinary device output accounting");
        accounting.host_output_staging_bytes =
            checked_add(accounting.host_output_staging_bytes, stats.host_output_staging_bytes,
                        "Runtime host staging accounting");
        any_cuda_graph_active = any_cuda_graph_active || stats.cuda_graph_active;

        if (stats.engine_identity == 0) {
            all_engine_identities_available = false;
            all_engine_weights_available = false;
            all_resident_weights_available = false;
            continue;
        }
        const auto [found, inserted] = engines.emplace(stats.engine_identity, stats);
        if (!inserted) {
            const auto& previous = found->second;
            if (previous.total_weight_bytes != stats.total_weight_bytes ||
                previous.streamable_weight_bytes != stats.streamable_weight_bytes ||
                previous.weight_streaming_budget_bytes != stats.weight_streaming_budget_bytes ||
                previous.total_weight_bytes_available != stats.total_weight_bytes_available ||
                previous.weight_streaming_budget_available !=
                    stats.weight_streaming_budget_available) {
                throw std::runtime_error(
                    "Profile contexts sharing an engine reported inconsistent weight stats");
            }
        }
    }

    for (const auto& [identity, stats] : engines) {
        (void)identity;
        if (!stats.total_weight_bytes_available) {
            all_engine_weights_available = false;
            all_resident_weights_available = false;
            continue;
        }
        accounting.engine_weight_bytes =
            checked_add(accounting.engine_weight_bytes, stats.total_weight_bytes,
                        "Runtime engine weight accounting");

        const bool streaming_active =
            stats.streamable_weight_bytes > 0 &&
            (!stats.weight_streaming_budget_available ||
             stats.weight_streaming_budget_bytes < stats.streamable_weight_bytes);
        accounting.weight_streaming_active = accounting.weight_streaming_active || streaming_active;
        if (streaming_active) {
            // TensorRT explicitly documents TOTAL_WEIGHTS_SIZE as not equal to
            // resident device bytes under weight streaming. Do not invent a
            // sub-allocation from the streaming budget.
            all_resident_weights_available = false;
        } else {
            accounting.resident_weight_bytes =
                checked_add(accounting.resident_weight_bytes, stats.total_weight_bytes,
                            "Runtime resident weight accounting");
        }
    }

    accounting.engine_weight_bytes_available =
        all_introspection_available && all_engine_weights_available;
    accounting.resident_weight_copy_count_available =
        all_introspection_available && all_engine_identities_available;
    if (accounting.resident_weight_copy_count_available) {
        if (engines.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("Runtime resident weight copy count overflows uint32");
        accounting.resident_weight_copy_count = static_cast<std::uint32_t>(engines.size());
    }
    accounting.resident_weight_bytes_available =
        accounting.engine_weight_bytes_available && all_resident_weights_available;
    accounting.module_allocation_bytes_available = all_introspection_available;
    accounting.cuda_graph_private_bytes_available =
        all_introspection_available && !any_cuda_graph_active;
    return accounting;
}

std::vector<const RuntimeKvExecutionRole*> enabled_roles(const RuntimeKvSetupRequest& request,
                                                         std::uint64_t capacity) {
    std::vector<const RuntimeKvExecutionRole*> result;
    result.reserve(request.roles.size());

    const RuntimeKvExecutionRole* prefill = nullptr;
    std::vector<const RuntimeKvExecutionRole*> decode;
    for (const auto& role : request.roles) {
        if (role.kind == RuntimeKvExecutionRoleKind::kPrefill) {
            if (prefill != nullptr)
                throw std::invalid_argument(
                    "Runtime KV setup currently supports one prefill execution role");
            prefill = &role;
        } else {
            decode.push_back(&role);
        }
    }
    std::sort(decode.begin(), decode.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->profile_limit < rhs->profile_limit;
    });
    if (decode.empty() || decode.back()->profile_limit < capacity) {
        throw std::runtime_error("No decode optimization profile covers runtime KV capacity " +
                                 std::to_string(capacity));
    }

    result.push_back(prefill);
    for (const auto* role : decode) {
        result.push_back(role);
        if (role->profile_limit >= capacity)
            break;
    }
    return result;
}

struct ContextEnvelope {
    RuntimeMemoryOverhead overhead;
    std::size_t alignment{kRuntimeMemoryCudaAlignmentV1};
    int32_t device{-1};
};

std::uint64_t history_bound_for_capacity(const RuntimeKvSetupRequest& request,
                                         std::uint64_t capacity, std::uint64_t history_tokens) {
    if (history_tokens == 0)
        return 1;
    const auto found =
        std::lower_bound(request.expected_active_kv_profile_limits.begin(),
                         request.expected_active_kv_profile_limits.end(), history_tokens);
    if (found == request.expected_active_kv_profile_limits.end()) {
        throw std::runtime_error("No history-bound profile covers runtime KV history " +
                                 std::to_string(history_tokens));
    }
    return std::min(*found, capacity);
}

void include_context_requirement(ContextEnvelope& envelope, const RuntimeKvExecutionRole& role,
                                 const RuntimeKvGraphLayout& layout, std::uint64_t history_tokens,
                                 std::uint64_t query_tokens, std::uint64_t bound_tokens) {
    const auto requirement = plan_runtime_kv_invocation(*role.module, layout, history_tokens,
                                                        query_tokens, bound_tokens);
    if (requirement.struct_size != sizeof(RuntimeMemoryContextRequirementV1) ||
        requirement.api_version != kRuntimeMemoryBackendApiVersionV1) {
        throw std::runtime_error("Runtime KV context requirement has an incompatible ABI");
    }
    if (requirement.device < 0) {
        throw std::runtime_error("Runtime KV context requirement has an invalid CUDA device");
    }
    if (requirement.alignment == 0 || (requirement.alignment & (requirement.alignment - 1)) != 0) {
        throw std::runtime_error("Runtime KV context requirement has an invalid alignment");
    }
    if (envelope.device < 0) {
        envelope.device = requirement.device;
    } else if (envelope.device != requirement.device) {
        throw std::runtime_error("Runtime KV execution roles reside on different CUDA devices");
    }
    envelope.alignment = std::max(envelope.alignment, requirement.alignment);
    envelope.overhead.context_device_memory_bytes =
        std::max(envelope.overhead.context_device_memory_bytes,
                 static_cast<std::uint64_t>(requirement.capacity_bytes));
}

ContextEnvelope query_context_envelope(const RuntimeKvSetupRequest& request,
                                       std::uint64_t capacity) {
    auto layout = request.layout;
    layout.capacity_tokens = capacity;
    layout.active_kv_profile_limits = request.expected_active_kv_profile_limits;

    ContextEnvelope envelope;
    envelope.overhead.external_device_output_bytes = staging_bytes(request, capacity);
    const auto roles = enabled_roles(request, capacity);
    for (const auto* role_ptr : roles) {
        const auto& role = *role_ptr;
        if (role.kind == RuntimeKvExecutionRoleKind::kPrefill) {
            // TensorRT does not guarantee that USER_MANAGED context memory is
            // monotonic between a profile's endpoints. Probe every Sq for
            // every history bucket reachable by the native chunk scheduler,
            // then retain the true maximum. H itself is a scalar value; the
            // TensorRT shape is determined by Sq and the padded history T.
            const auto cold_query_limit =
                std::min<std::uint64_t>(layout.prefill_chunk_limit, capacity);
            for (std::uint64_t query_tokens = 1; query_tokens <= cold_query_limit; ++query_tokens) {
                include_context_requirement(envelope, role, layout, /*history_tokens=*/0,
                                            query_tokens, /*bound_tokens=*/1);
            }

            std::set<std::uint64_t> probed_history_bounds;
            const auto chunk = layout.prefill_chunk_limit;
            for (std::uint64_t history_tokens = chunk; history_tokens < capacity;) {
                const auto bound_tokens =
                    history_bound_for_capacity(request, capacity, history_tokens);
                if (probed_history_bounds.insert(bound_tokens).second) {
                    const auto query_limit =
                        std::min<std::uint64_t>(chunk, capacity - history_tokens);
                    for (std::uint64_t query_tokens = 1; query_tokens <= query_limit;
                         ++query_tokens) {
                        include_context_requirement(envelope, role, layout, history_tokens,
                                                    query_tokens, bound_tokens);
                    }
                }
                if (chunk > capacity - history_tokens)
                    break;
                history_tokens += chunk;
            }
            continue;
        }

        const auto bound_tokens = std::min(capacity, role.profile_limit);
        // The smallest decode role is also the cold-sentinel fallback.
        if (role.profile_limit == request.expected_active_kv_profile_limits.front()) {
            include_context_requirement(envelope, role, layout, /*history_tokens=*/0,
                                        /*query_tokens=*/1, /*bound_tokens=*/1);
        }
        const auto found =
            std::lower_bound(request.expected_active_kv_profile_limits.begin(),
                             request.expected_active_kv_profile_limits.end(), role.profile_limit);
        const auto previous_limit =
            found == request.expected_active_kv_profile_limits.begin() ? 0 : *(found - 1);
        const auto history_tokens = previous_limit + 1;
        if (history_tokens < capacity) {
            include_context_requirement(envelope, role, layout, history_tokens,
                                        /*query_tokens=*/1, bound_tokens);
        }
    }
    // Shape planning above materializes ordinary dynamic I/O. Sum every
    // actual module/context allocation once, including high-water capacities
    // retained by a role that a later decreasing solve no longer enables.
    const auto module_accounting = query_engine_accounting(request.roles);
    envelope.overhead.ordinary_device_input_bytes = module_accounting.ordinary_device_input_bytes;
    envelope.overhead.ordinary_device_output_bytes = module_accounting.ordinary_device_output_bytes;
    return envelope;
}

std::unique_ptr<RuntimeKvAllocation> allocate_staging(const RuntimeKvSetupRequest& request,
                                                      std::uint64_t capacity,
                                                      std::uint32_t device) {
    return std::make_unique<RuntimeKvAllocation>(
        request.layout.layer_count, staging_capacity_tokens(request.layout, capacity),
        checked_mul(request.layout.kv_head_count, request.layout.head_dim,
                    "Runtime KV staging row width"),
        dtype_size(request.layout.dtype), device, request.stream, request.allocator);
}

void synchronize_overhead_release(const RuntimeKvSetupRequest& request) {
    // Deterministic unit tests inject a synchronous host allocator together
    // with a synthetic memory query. Production CUDA allocations must make an
    // async context/staging release visible before replacement allocations.
    if (request.query_device_memory)
        return;
    const auto status = request.stream != nullptr
                            ? cudaStreamSynchronize(static_cast<cudaStream_t>(request.stream))
                            : cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize resized runtime KV overhead: " +
                                 std::string(cudaGetErrorString(status)));
    }
}

RuntimeMemoryContextBlockV1 allocate_context_block(const RuntimeKvSetupRequest& request,
                                                   const ContextEnvelope& envelope,
                                                   std::uint32_t device) {
    if (envelope.device != static_cast<int32_t>(device)) {
        throw std::runtime_error(
            "Runtime KV context device does not match the selected CUDA device");
    }
    RuntimeMemoryContextBlockV1 block;
    block.capacity_bytes = static_cast<std::size_t>(envelope.overhead.context_device_memory_bytes);
    block.alignment = envelope.alignment;
    block.device = static_cast<int32_t>(device);
    if (block.capacity_bytes == 0)
        return block;

    auto allocation =
        request.allocator->allocate(block.capacity_bytes, block.alignment, device, request.stream);
    if (!allocation.valid() || allocation.bytes != block.capacity_bytes) {
        throw std::runtime_error("Runtime allocator returned an invalid shared context block");
    }
    if (allocation.device != device) {
        throw std::runtime_error(
            "Runtime allocator returned the shared context block on the wrong device");
    }
    if (allocation.alignment < block.alignment ||
        (reinterpret_cast<std::uintptr_t>(allocation.pointer) % block.alignment) != 0) {
        throw std::runtime_error(
            "Runtime allocator returned an under-aligned shared context block");
    }
    block.pointer = allocation.pointer;
    block.capacity_bytes = static_cast<std::size_t>(allocation.bytes);
    block.alignment = static_cast<std::size_t>(allocation.alignment);
    block.device = static_cast<int32_t>(allocation.device);
    block.lifetime = std::move(allocation.owner);
    return block;
}

} // namespace

RuntimeDeviceMemorySnapshot query_runtime_device_memory_snapshot(const char* phase) {
    if (runtime_device_memory_qualification_pre_snapshot_action)
        runtime_device_memory_qualification_pre_snapshot_action(phase);
    const auto snapshot = cuda_memory_snapshot_after_sync(phase);
    if (runtime_device_memory_qualification_observer)
        runtime_device_memory_qualification_observer(phase, snapshot);
    return snapshot;
}

void set_runtime_device_memory_qualification_observer(
    RuntimeDeviceMemoryQualificationObserver observer) {
    runtime_device_memory_qualification_observer = std::move(observer);
}

void set_runtime_device_memory_qualification_pre_snapshot_action(
    RuntimeDeviceMemoryQualificationPreSnapshotAction action) {
    runtime_device_memory_qualification_pre_snapshot_action = std::move(action);
}

std::unique_ptr<RuntimeKvStateCore> create_runtime_kv_state(const RuntimeKvSetupRequest& request) {
    validate_request(request);

    const auto query_device_memory =
        request.query_device_memory
            ? request.query_device_memory
            : RuntimeDeviceMemoryQuery{query_runtime_device_memory_snapshot};
    const auto post_load = query_device_memory("before runtime KV planning");
    if (post_load.free_bytes == 0 || post_load.total_bytes == 0 ||
        post_load.free_bytes > post_load.total_bytes) {
        throw std::runtime_error("Runtime KV setup received an invalid post-load memory snapshot");
    }

    RuntimeMemoryPlanRequest plan_request;
    plan_request.post_load_free_bytes = post_load.free_bytes;
    plan_request.safety_reserve_bytes = request.safety_reserve_bytes;
    plan_request.kv_bytes_per_token = request.expected_kv_bytes_per_token;
    plan_request.model_context_limit = request.layout.capacity_tokens;
    plan_request.request_context_limit = request.request_context_limit;
    plan_request.prefill_chunk_limit = request.layout.prefill_chunk_limit;
    plan_request.policy = request.policy;
    plan_request.active_kv_profile_limits = request.expected_active_kv_profile_limits;
    plan_request.module_residency_reserve_bytes_by_profile =
        request.module_residency_reserve_bytes_by_profile;

    const auto query = [&](std::uint64_t capacity, const std::vector<std::uint64_t>&) {
        return query_context_envelope(request, capacity).overhead;
    };
    auto plan = solve_runtime_memory_plan(plan_request, query);
    const auto context_envelope = query_context_envelope(request, plan.runtime_kv_capacity_tokens);
    auto context_block = allocate_context_block(request, context_envelope, post_load.device);
    auto staging = allocate_staging(request, plan.runtime_kv_capacity_tokens, post_load.device);
    if (staging->total_bytes() != context_envelope.overhead.external_device_output_bytes) {
        throw std::runtime_error("Runtime KV staging size does not match the resolved memory plan");
    }

    const auto capacity_decision_snapshot =
        query_device_memory("after shared context and output allocation");
    if (capacity_decision_snapshot.device != post_load.device) {
        throw std::runtime_error("CUDA device changed during runtime KV memory setup");
    }
    if (capacity_decision_snapshot.free_bytes == 0 || capacity_decision_snapshot.total_bytes == 0 ||
        capacity_decision_snapshot.free_bytes > capacity_decision_snapshot.total_bytes ||
        capacity_decision_snapshot.total_bytes != post_load.total_bytes) {
        throw std::runtime_error(
            "Runtime KV setup received an invalid capacity-decision memory snapshot");
    }
    plan_request.capacity_decision_free_bytes = capacity_decision_snapshot.free_bytes;
    plan_request.capacity_decision_upper_bound_tokens = plan.runtime_kv_capacity_tokens;
    plan_request.capacity_decision_resident_overhead = context_envelope.overhead;
    auto final_plan = solve_runtime_memory_plan(plan_request, query);
    const auto final_envelope =
        query_context_envelope(request, final_plan.runtime_kv_capacity_tokens);
    if (final_plan.runtime_kv_capacity_tokens > plan.runtime_kv_capacity_tokens) {
        throw std::runtime_error("Runtime KV final memory solve attempted to increase capacity");
    }
    if (final_envelope.device != context_block.device) {
        throw std::runtime_error(
            "Runtime KV final context envelope moved to a different CUDA device");
    }
    if (final_envelope.overhead.external_device_output_bytes > staging->total_bytes()) {
        throw std::runtime_error("Runtime KV final memory solve attempted to increase staging");
    }

    const bool resize_context =
        final_envelope.overhead.context_device_memory_bytes != context_block.capacity_bytes ||
        final_envelope.alignment > context_block.alignment ||
        (context_block.pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(context_block.pointer) % final_envelope.alignment != 0);
    const bool resize_staging =
        staging->total_bytes() != final_envelope.overhead.external_device_output_bytes;
    if (resize_context || resize_staging) {
        // The second, post-overhead observation may only decrease R. Release
        // every replaced O(R) reservation first so cudaMallocAsync cannot
        // transiently retain both the initial and final envelopes. A lower R
        // can require a larger context block at a tactic discontinuity.
        if (resize_context)
            context_block = RuntimeMemoryContextBlockV1{};
        if (resize_staging)
            staging.reset();
        synchronize_overhead_release(request);
    }
    if (resize_context) {
        context_block = allocate_context_block(request, final_envelope, post_load.device);
        if (context_block.capacity_bytes != final_envelope.overhead.context_device_memory_bytes) {
            throw std::runtime_error(
                "Runtime KV final context size does not match its memory plan");
        }
    }
    if (resize_staging) {
        staging =
            allocate_staging(request, final_plan.runtime_kv_capacity_tokens, post_load.device);
        if (staging->total_bytes() != final_envelope.overhead.external_device_output_bytes) {
            throw std::runtime_error(
                "Runtime KV final staging size does not match its memory plan");
        }
    }

    auto layout = request.layout;
    layout.capacity_tokens = final_plan.runtime_kv_capacity_tokens;
    layout.active_kv_profile_limits = request.expected_active_kv_profile_limits;
    auto allocation = std::make_unique<RuntimeKvAllocation>(
        layout.layer_count, layout.capacity_tokens,
        checked_mul(layout.kv_head_count, layout.head_dim, "Runtime KV allocation row width"),
        dtype_size(layout.dtype), post_load.device, request.stream, request.allocator);
    if (allocation->total_bytes() != final_plan.allocated_kv_bytes) {
        throw std::runtime_error(
            "Runtime KV allocation size does not match the resolved memory plan");
    }

    auto receipt = final_plan.receipt;
    receipt.module_residency_plan_set_sha256 =
        request.module_residency_plan_set_sha256;
    receipt.module_residency_cuda_module_loading_mode =
        request.module_residency_cuda_module_loading_mode;
    receipt.module_residency_evidence_sha256 =
        request.module_residency_evidence_sha256;
    receipt.serialized_plan_bytes = request.serialized_plan_bytes;
    if (request.pre_load_memory_snapshot_available) {
        const auto& pre_load = request.pre_load_memory_snapshot;
        if (pre_load.device != post_load.device || pre_load.total_bytes == 0 ||
            pre_load.free_bytes == 0 || pre_load.free_bytes > pre_load.total_bytes) {
            throw std::runtime_error(
                "Runtime KV setup received an invalid pre-load memory snapshot");
        }
        receipt.pre_load_free_bytes = pre_load.free_bytes;
        receipt.pre_load_total_bytes = pre_load.total_bytes;
        receipt.pre_load_snapshot_available = true;
    }
    receipt.post_load_total_bytes = post_load.total_bytes;
    receipt.post_load_device_used_bytes = post_load.total_bytes - post_load.free_bytes;
    receipt.post_load_total_bytes_available = true;
    receipt.capacity_decision_free_bytes = capacity_decision_snapshot.free_bytes;
    receipt.capacity_decision_total_bytes = capacity_decision_snapshot.total_bytes;
    receipt.capacity_decision_device_used_bytes =
        capacity_decision_snapshot.total_bytes - capacity_decision_snapshot.free_bytes;
    receipt.capacity_decision_snapshot_available = true;
    // Schema-v2 compatibility: final_* was historically the snapshot used to
    // make the final capacity decision. Keep that binding explicit while
    // schema-v4 consumers use settled_* for actual post-allocation residency.
    receipt.final_free_bytes = receipt.capacity_decision_free_bytes;
    receipt.final_total_bytes = receipt.capacity_decision_total_bytes;
    receipt.final_device_used_bytes = receipt.capacity_decision_device_used_bytes;
    receipt.final_snapshot_available = true;
    receipt.context_device_memory_bytes = final_plan.overhead.context_device_memory_bytes;
    const auto engine_accounting = query_engine_accounting(request.roles);
    receipt.engine_weight_bytes = engine_accounting.engine_weight_bytes;
    receipt.engine_weight_bytes_available = engine_accounting.engine_weight_bytes_available;
    receipt.resident_weight_bytes = engine_accounting.resident_weight_bytes;
    receipt.resident_weight_bytes_available = engine_accounting.resident_weight_bytes_available;
    receipt.resident_weight_copy_count = engine_accounting.resident_weight_copy_count;
    receipt.resident_weight_copy_count_available =
        engine_accounting.resident_weight_copy_count_available;
    receipt.weight_streaming_active = engine_accounting.weight_streaming_active;
    receipt.ordinary_device_input_bytes = engine_accounting.ordinary_device_input_bytes;
    receipt.ordinary_device_output_bytes = engine_accounting.ordinary_device_output_bytes;
    receipt.ordinary_device_input_bytes_available =
        engine_accounting.module_allocation_bytes_available;
    receipt.ordinary_device_output_bytes_available =
        engine_accounting.module_allocation_bytes_available;
    receipt.external_device_output_bytes = staging->total_bytes();
    receipt.host_staging_bytes = engine_accounting.host_output_staging_bytes;
    receipt.external_device_output_bytes_available = true;
    receipt.host_staging_bytes_available = engine_accounting.module_allocation_bytes_available;
    receipt.graph_private_device_bytes = 0;
    receipt.graph_private_device_bytes_available =
        engine_accounting.cuda_graph_private_bytes_available;
    try {
        const auto load_complete = query_device_memory("after runtime KV allocation");
        if (load_complete.device != post_load.device || load_complete.free_bytes == 0 ||
            load_complete.total_bytes == 0 ||
            load_complete.free_bytes > load_complete.total_bytes ||
            load_complete.total_bytes != post_load.total_bytes) {
            receipt.settled_snapshot_available = false;
            receipt.settled_snapshot_unavailable_reason = "settled_cuda_memory_snapshot_invalid";
            receipt.mark_peak_device_sampling_failed(
                "load_completion_cuda_memory_snapshot_invalid");
        } else {
            receipt.settled_free_bytes = load_complete.free_bytes;
            receipt.settled_total_bytes = load_complete.total_bytes;
            receipt.settled_device_used_bytes =
                load_complete.total_bytes - load_complete.free_bytes;
            receipt.settled_snapshot_available = true;
            receipt.settled_snapshot_unavailable_reason.clear();
            if (!receipt.pre_load_snapshot_available) {
                receipt.mark_peak_device_sampling_failed(
                    "pre_load_cuda_memory_snapshot_unavailable");
            } else {
                receipt.observe_peak_device_memory(
                    load_complete.free_bytes, RuntimeMemoryPeakSampleBoundary::kLoadCompletion);
            }
        }
    } catch (...) {
        // The cache is already valid. Preserve inference correctness and make
        // the missing settled/high-water observability explicit in the receipt.
        receipt.settled_snapshot_available = false;
        receipt.settled_snapshot_unavailable_reason = "settled_cuda_mem_get_info_failed";
        receipt.mark_peak_device_sampling_failed("load_completion_cuda_mem_get_info_failed");
    }
    return std::make_unique<RuntimeKvStateCore>(
        std::move(layout), std::move(allocation), std::move(staging), std::move(context_block),
        std::move(receipt), request.stream, request.device_copy, request.stream_synchronize);
}

} // namespace trtmc
