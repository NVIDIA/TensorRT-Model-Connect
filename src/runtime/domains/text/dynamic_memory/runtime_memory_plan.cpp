/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_memory_plan.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc {

namespace {

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
        throw std::overflow_error(std::string(what) + " byte count overflows uint64");
    return lhs + rhs;
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
        throw std::overflow_error(std::string(what) + " byte count overflows uint64");
    return lhs * rhs;
}

std::uint64_t semantic_limit(const RuntimeMemoryPlanRequest& request) {
    std::uint64_t limit = request.model_context_limit;
    if (request.request_context_limit != 0)
        limit = std::min(limit, request.request_context_limit);
    return limit;
}

std::vector<std::uint64_t> enabled_profiles(const RuntimeMemoryPlanRequest& request,
                                            std::uint64_t capacity) {
    std::vector<std::uint64_t> enabled;
    enabled.reserve(request.active_kv_profile_limits.size());
    for (const auto limit : request.active_kv_profile_limits) {
        if (limit == 0 || limit > request.model_context_limit)
            throw std::invalid_argument(
                "Active-KV profile limits must be in [1, model_context_limit]");
        if (limit <= capacity)
            enabled.push_back(limit);
    }
    if (enabled.empty() || enabled.back() != capacity)
        enabled.push_back(capacity);
    std::sort(enabled.begin(), enabled.end());
    enabled.erase(std::unique(enabled.begin(), enabled.end()), enabled.end());
    return enabled;
}

std::uint64_t bytes_after_fixed_reserve(std::uint64_t free_bytes, std::uint64_t safety_reserve) {
    if (free_bytes <= safety_reserve)
        return 0;
    return free_bytes - safety_reserve;
}

std::uint64_t fraction_budget_bytes(double fraction, std::uint64_t bytes) {
    const long double scaled = static_cast<long double>(bytes) * static_cast<long double>(fraction);
    return static_cast<std::uint64_t>(
        std::min(scaled, static_cast<long double>(std::numeric_limits<std::uint64_t>::max())));
}

std::uint64_t fraction_rows(double fraction, std::uint64_t bytes, std::uint64_t row_bytes) {
    return fraction_budget_bytes(fraction, bytes) / row_bytes;
}

std::uint64_t overhead_device_bytes(const RuntimeMemoryOverhead& overhead) {
    return overhead.device_bytes();
}

void validate_request(const RuntimeMemoryPlanRequest& request) {
    if (request.post_load_free_bytes == 0)
        throw std::invalid_argument("Runtime memory planning requires post-load free GPU memory");
    if (request.kv_bytes_per_token == 0)
        throw std::invalid_argument("KV bytes per token must be positive");
    if (request.model_context_limit == 0)
        throw std::invalid_argument("Model context limit must be positive");
    if (request.prefill_chunk_limit == 0 ||
        request.prefill_chunk_limit > request.model_context_limit) {
        throw std::invalid_argument("Prefill chunk limit must be in [1, model_context_limit]");
    }
    if (request.request_context_limit > request.model_context_limit) {
        throw std::invalid_argument(
            "Requested max sequence length " + std::to_string(request.request_context_limit) +
            " exceeds the model context limit " + std::to_string(request.model_context_limit));
    }
    if (request.max_solve_iterations == 0)
        throw std::invalid_argument("Runtime memory solve iteration limit must be positive");

    if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
        if (request.policy.bytes == 0)
            throw std::invalid_argument("Explicit KV byte policy must be positive");
    } else if (!(std::isfinite(request.policy.fraction) && request.policy.fraction > 0.0 &&
                 request.policy.fraction <= 1.0)) {
        throw std::invalid_argument("KV cache memory fraction must be in (0, 1]");
    }
}

RuntimeMemoryReceipt make_receipt(const RuntimeMemoryPlanRequest& request,
                                  const RuntimeMemoryPlan& plan, std::uint32_t iterations) {
    RuntimeMemoryReceipt receipt;
    receipt.policy = request.policy.kind;
    receipt.policy_fraction =
        request.policy.kind == RuntimeKvPolicyKind::kBytes ? 0.0 : request.policy.fraction;
    receipt.requested_kv_bytes =
        request.policy.kind == RuntimeKvPolicyKind::kBytes ? request.policy.bytes : 0;
    receipt.post_load_free_bytes = request.post_load_free_bytes;
    receipt.safety_reserve_bytes = request.safety_reserve_bytes;
    receipt.model_context_limit = request.model_context_limit;
    receipt.prefill_chunk_limit = request.prefill_chunk_limit;
    receipt.request_context_limit = request.request_context_limit;
    receipt.runtime_kv_capacity_tokens = plan.runtime_kv_capacity_tokens;
    receipt.effective_request_limit =
        std::min(semantic_limit(request), plan.runtime_kv_capacity_tokens);
    receipt.kv_bytes_per_token = request.kv_bytes_per_token;
    if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
        receipt.kv_budget_bytes = request.policy.bytes;
    } else if (request.final_free_bytes != 0) {
        receipt.kv_budget_bytes = fraction_budget_bytes(
            request.policy.fraction,
            bytes_after_fixed_reserve(request.final_free_bytes, request.safety_reserve_bytes));
    } else {
        const auto safe_free =
            bytes_after_fixed_reserve(request.post_load_free_bytes, request.safety_reserve_bytes);
        const auto overhead_bytes = plan.overhead.device_bytes();
        receipt.kv_budget_bytes = fraction_budget_bytes(
            request.policy.fraction, overhead_bytes < safe_free ? safe_free - overhead_bytes : 0);
    }
    receipt.context_device_memory_bytes = plan.overhead.context_device_memory_bytes;
    receipt.external_device_output_bytes = plan.overhead.external_device_output_bytes;
    receipt.host_staging_bytes = plan.overhead.host_staging_bytes;
    receipt.graph_private_device_bytes = plan.overhead.graph_private_device_bytes;
    receipt.kv_reserved_bytes = plan.allocated_kv_bytes;
    receipt.kv_committed_bytes = plan.allocated_kv_bytes;
    receipt.solve_iterations = iterations;
    receipt.capped_by_model = plan.runtime_kv_capacity_tokens == request.model_context_limit;
    receipt.capped_by_request_limit =
        request.request_context_limit != 0 &&
        plan.runtime_kv_capacity_tokens == request.request_context_limit;
    return receipt;
}

class CudaRuntimeDeviceAllocator final : public IRuntimeDeviceAllocator {
  public:
    RuntimeDeviceAllocation allocate(std::uint64_t bytes, std::uint64_t alignment,
                                     std::uint32_t device, void* stream) override {
        if (bytes == 0)
            throw std::invalid_argument("Runtime device allocation size must be positive");
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            throw std::invalid_argument("Runtime device allocation alignment must be a power of 2");

        int previous_device = 0;
        if (cudaGetDevice(&previous_device) != cudaSuccess)
            throw std::runtime_error("Unable to query the active CUDA device");
        if (cudaSetDevice(static_cast<int>(device)) != cudaSuccess)
            throw std::runtime_error("Unable to select CUDA device " + std::to_string(device));

        void* pointer = nullptr;
        cudaError_t status = cudaSuccess;
        if (stream != nullptr) {
            status = cudaMallocAsync(&pointer, static_cast<std::size_t>(bytes),
                                     static_cast<cudaStream_t>(stream));
        } else {
            status = cudaMalloc(&pointer, static_cast<std::size_t>(bytes));
        }
        const auto restore_status = cudaSetDevice(previous_device);
        if (status != cudaSuccess || restore_status != cudaSuccess) {
            if (pointer != nullptr)
                cudaFree(pointer);
            throw std::runtime_error("Unable to allocate " + std::to_string(bytes) +
                                     " runtime device bytes: " + cudaGetErrorString(status));
        }
        if (reinterpret_cast<std::uintptr_t>(pointer) % alignment != 0) {
            cudaFree(pointer);
            throw std::runtime_error("CUDA allocation does not satisfy requested alignment");
        }

        auto owner = std::shared_ptr<void>(pointer, [device, stream](void* allocation) {
            if (allocation == nullptr)
                return;
            int previous = 0;
            const bool restore = cudaGetDevice(&previous) == cudaSuccess;
            cudaSetDevice(static_cast<int>(device));
            if (stream != nullptr)
                cudaFreeAsync(allocation, static_cast<cudaStream_t>(stream));
            else
                cudaFree(allocation);
            if (restore)
                cudaSetDevice(previous);
        });
        return RuntimeDeviceAllocation{pointer, bytes, device, alignment, std::move(owner)};
    }
};

} // namespace

std::uint64_t RuntimeMemoryOverhead::device_bytes() const {
    auto total = checked_add(context_device_memory_bytes, external_device_output_bytes,
                             "Runtime device overhead");
    return checked_add(total, graph_private_device_bytes, "Runtime device overhead");
}

const char* runtime_kv_policy_name(RuntimeKvPolicyKind policy) {
    switch (policy) {
    case RuntimeKvPolicyKind::kAuto:
        return "auto";
    case RuntimeKvPolicyKind::kFraction:
        return "fraction";
    case RuntimeKvPolicyKind::kBytes:
        return "bytes";
    }
    return "unknown";
}

void RuntimeMemoryReceipt::observe_peak_device_memory(
    std::uint64_t current_free_bytes, RuntimeMemoryPeakSampleBoundary boundary) noexcept {
    if (peak_device_sampling_failed)
        return;
    if (!pre_load_snapshot_available || pre_load_free_bytes == 0 || current_free_bytes == 0) {
        mark_peak_device_sampling_failed("pre_load_or_sampled_free_memory_unavailable");
        return;
    }

    const auto device_delta = pre_load_free_bytes > current_free_bytes
                                  ? pre_load_free_bytes - current_free_bytes
                                  : std::uint64_t{0};
    peak_device_bytes = std::max(peak_device_bytes, device_delta);
    peak_device_bytes_available = true;
    peak_device_bytes_unavailable_reason.clear();
    ++peak_device_sample_count;
    if (boundary == RuntimeMemoryPeakSampleBoundary::kLoadCompletion)
        peak_sampled_at_load_completion = true;
    else
        peak_sampled_at_request_completion = true;
}

void RuntimeMemoryReceipt::mark_peak_device_sampling_failed(const char* reason) noexcept {
    peak_device_sampling_failed = true;
    peak_device_bytes_available = false;
    peak_device_bytes_unavailable_reason =
        reason != nullptr && reason[0] != '\0' ? reason : "cuda_memory_high_water_sampling_failed";
}

RuntimeMemoryPlan solve_runtime_memory_plan(const RuntimeMemoryPlanRequest& request,
                                            const RuntimeOverheadQuery& query_overhead) {
    validate_request(request);
    if (!query_overhead)
        throw std::invalid_argument("Runtime memory planning requires an overhead query");

    const auto useful_limit = semantic_limit(request);
    const auto safe_free =
        bytes_after_fixed_reserve(request.post_load_free_bytes, request.safety_reserve_bytes);
    if (safe_free < request.kv_bytes_per_token)
        throw std::runtime_error("Safely usable GPU memory is smaller than one KV token row");

    std::uint64_t candidate = 0;
    if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
        candidate = std::min(useful_limit, request.policy.bytes / request.kv_bytes_per_token);
    } else {
        candidate = std::min(useful_limit, fraction_rows(request.policy.fraction, safe_free,
                                                         request.kv_bytes_per_token));
    }
    if (candidate == 0)
        throw std::runtime_error("KV cache memory budget is smaller than one token row (" +
                                 std::to_string(request.kv_bytes_per_token) + " bytes)");

    RuntimeMemoryPlan plan;
    std::uint32_t iterations = 0;
    if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
        plan.enabled_profile_limits = enabled_profiles(request, candidate);
        plan.overhead = query_overhead(candidate, plan.enabled_profile_limits);
        const auto kv_bytes = checked_mul(candidate, request.kv_bytes_per_token, "KV cache");
        const auto required =
            checked_add(kv_bytes, overhead_device_bytes(plan.overhead), "Runtime allocation");
        if (required > safe_free) {
            throw std::runtime_error("Explicit KV policy resolves to " + std::to_string(kv_bytes) +
                                     " KV bytes plus " +
                                     std::to_string(overhead_device_bytes(plan.overhead)) +
                                     " non-KV device bytes, exceeding " +
                                     std::to_string(safe_free) + " safely available bytes");
        }
        plan.runtime_kv_capacity_tokens = candidate;
        plan.allocated_kv_bytes = kv_bytes;
        iterations = 1;
    } else {
        for (; iterations < request.max_solve_iterations; ++iterations) {
            auto profiles = enabled_profiles(request, candidate);
            const auto overhead = query_overhead(candidate, profiles);
            const auto overhead_bytes = overhead_device_bytes(overhead);
            const auto remaining = overhead_bytes < safe_free ? safe_free - overhead_bytes : 0;
            const auto next = std::min(
                candidate, std::min(useful_limit, fraction_rows(request.policy.fraction, remaining,
                                                                request.kv_bytes_per_token)));
            plan.overhead = overhead;
            plan.enabled_profile_limits = std::move(profiles);
            if (next == candidate) {
                ++iterations;
                break;
            }
            if (next == 0)
                throw std::runtime_error(
                    "Runtime overhead leaves less than one KV token row available");
            candidate = next;
        }

        // Re-query the converged candidate. This also makes a bounded solve
        // deterministic when crossing a TensorRT profile/tactic discontinuity.
        plan.enabled_profile_limits = enabled_profiles(request, candidate);
        plan.overhead = query_overhead(candidate, plan.enabled_profile_limits);
        const auto overhead_bytes = overhead_device_bytes(plan.overhead);
        const auto remaining = overhead_bytes < safe_free ? safe_free - overhead_bytes : 0;
        auto fit = std::min(candidate, fraction_rows(request.policy.fraction, remaining,
                                                     request.kv_bytes_per_token));
        if (fit == 0)
            throw std::runtime_error(
                "Runtime overhead leaves less than one KV token row available");
        if (fit < candidate) {
            candidate = fit;
            plan.enabled_profile_limits = enabled_profiles(request, candidate);
            plan.overhead = query_overhead(candidate, plan.enabled_profile_limits);
        }
        plan.runtime_kv_capacity_tokens = candidate;
        plan.allocated_kv_bytes = checked_mul(candidate, request.kv_bytes_per_token, "KV cache");
    }

    if (request.final_free_bytes != 0) {
        const auto final_safe =
            bytes_after_fixed_reserve(request.final_free_bytes, request.safety_reserve_bytes);
        // final_free_bytes is observed *after* the selected context/output
        // overhead has been allocated. Do not charge O(R) twice here.
        if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
            if (plan.allocated_kv_bytes > final_safe) {
                throw std::runtime_error(
                    "Available GPU memory changed before the explicit KV allocation: resolved " +
                    std::to_string(plan.allocated_kv_bytes) + " KV bytes, now safely available " +
                    std::to_string(final_safe) + " bytes");
            }
        } else {
            const auto final_rows =
                fraction_rows(request.policy.fraction, final_safe, request.kv_bytes_per_token);
            const auto reduced = std::min(plan.runtime_kv_capacity_tokens, final_rows);
            if (reduced == 0) {
                throw std::runtime_error(
                    "Available GPU memory changed before KV allocation and now "
                    "leaves less than one token row");
            }
            plan.runtime_kv_capacity_tokens = reduced;
            plan.allocated_kv_bytes = checked_mul(reduced, request.kv_bytes_per_token, "KV cache");
            plan.enabled_profile_limits = enabled_profiles(request, reduced);
            plan.overhead = query_overhead(reduced, plan.enabled_profile_limits);
        }
    }

    plan.receipt = make_receipt(request, plan, iterations);
    return plan;
}

std::string RuntimeMemoryReceipt::to_json() const {
    std::ostringstream out;
    const auto nullable_u64 = [&out](const char* name, std::uint64_t value, bool available) {
        out << ",\"" << name << "\":";
        if (available)
            out << value;
        else
            out << "null";
    };
    const auto nullable_source = [&out](const char* name, const char* source, bool available) {
        out << "\"" << name << "\":";
        if (available)
            out << "\"" << source << "\"";
        else
            out << "null";
    };

    out << "{\"receipt_schema_version\":" << receipt_schema_version
        << ",\"contract_version\":" << contract_version << ",\"policy\":\""
        << runtime_kv_policy_name(policy) << "\",\"policy_fraction\":" << policy_fraction
        << ",\"requested_kv_bytes\":" << requested_kv_bytes
        << ",\"post_load_free_bytes\":" << post_load_free_bytes
        << ",\"safety_reserve_bytes\":" << safety_reserve_bytes
        << ",\"model_context_limit\":" << model_context_limit
        << ",\"prefill_chunk_limit\":" << prefill_chunk_limit
        << ",\"request_context_limit\":" << request_context_limit
        << ",\"runtime_kv_capacity_tokens\":" << runtime_kv_capacity_tokens
        << ",\"effective_request_limit\":" << effective_request_limit
        << ",\"kv_bytes_per_token\":" << kv_bytes_per_token
        << ",\"kv_budget_bytes\":" << kv_budget_bytes;
    nullable_u64("pre_load_free_bytes", pre_load_free_bytes, pre_load_snapshot_available);
    nullable_u64("pre_load_total_bytes", pre_load_total_bytes, pre_load_snapshot_available);
    out << ",\"serialized_plan_bytes\":" << serialized_plan_bytes;
    nullable_u64("resident_weight_bytes", resident_weight_bytes, resident_weight_bytes_available);
    nullable_u64("resident_weight_copy_count", resident_weight_copy_count,
                 resident_weight_copy_count_available);
    nullable_u64("engine_weight_bytes", engine_weight_bytes, engine_weight_bytes_available);
    out << ",\"weight_streaming_active\":" << (weight_streaming_active ? "true" : "false");
    nullable_u64("post_load_total_bytes", post_load_total_bytes, post_load_total_bytes_available);
    nullable_u64("post_load_device_used_bytes", post_load_device_used_bytes,
                 post_load_total_bytes_available);
    nullable_u64("final_free_bytes", final_free_bytes, final_snapshot_available);
    nullable_u64("final_total_bytes", final_total_bytes, final_snapshot_available);
    nullable_u64("final_device_used_bytes", final_device_used_bytes, final_snapshot_available);
    out << ",\"context_device_memory_bytes\":" << context_device_memory_bytes;
    nullable_u64("external_device_output_bytes", external_device_output_bytes,
                 external_device_output_bytes_available);
    nullable_u64("host_staging_bytes", host_staging_bytes, host_staging_bytes_available);
    nullable_u64("graph_private_device_bytes", graph_private_device_bytes,
                 graph_private_device_bytes_available);
    out << ",\"kv_reserved_bytes\":" << kv_reserved_bytes
        << ",\"kv_committed_bytes\":" << kv_committed_bytes;
    nullable_u64("kv_metadata_bytes", kv_metadata_bytes, kv_metadata_bytes_available);
    nullable_u64("peak_device_bytes", peak_device_bytes, peak_device_bytes_available);
    out << ",\"peak_device_bytes_scope\":\"device_wide\""
        << ",\"peak_device_bytes_baseline\":"
           "\"cuda_mem_get_info_before_engine_deserialization_free\""
        << ",\"peak_device_sample_count\":" << peak_device_sample_count
        << ",\"peak_device_sample_boundaries\":[";
    bool wrote_peak_boundary = false;
    if (peak_sampled_at_load_completion) {
        out << "\"after_runtime_kv_allocation\"";
        wrote_peak_boundary = true;
    }
    if (peak_sampled_at_request_completion) {
        if (wrote_peak_boundary)
            out << ',';
        out << "\"after_successful_request_completion\"";
    }
    out << ']';
    out << ",\"backend_owned_cache_input_bytes\":" << backend_owned_cache_input_bytes
        << ",\"backend_owned_cache_output_bytes\":" << backend_owned_cache_output_bytes
        << ",\"kv_allocation_id\":" << kv_allocation_id
        << ",\"solve_iterations\":" << solve_iterations
        << ",\"capped_by_model\":" << (capped_by_model ? "true" : "false")
        << ",\"capped_by_request_limit\":" << (capped_by_request_limit ? "true" : "false")
        << ",\"measurement_sources\":{";
    nullable_source("pre_load_free_bytes", "cuda_mem_get_info_before_engine_deserialization",
                    pre_load_snapshot_available);
    out << ',';
    nullable_source("pre_load_total_bytes", "cuda_mem_get_info_device_total",
                    pre_load_snapshot_available);
    out << ',';
    nullable_source("post_load_free_bytes", "cuda_mem_get_info_after_engine_deserialization", true);
    out << ',';
    nullable_source("post_load_total_bytes", "cuda_mem_get_info_device_total",
                    post_load_total_bytes_available);
    out << ',';
    nullable_source("post_load_device_used_bytes",
                    "cuda_mem_get_info_device_total_minus_free_device_wide",
                    post_load_total_bytes_available);
    out << ',';
    nullable_source("final_free_bytes", "cuda_mem_get_info_after_context_and_output_reservation",
                    final_snapshot_available);
    out << ',';
    nullable_source("final_total_bytes", "cuda_mem_get_info_device_total",
                    final_snapshot_available);
    out << ',';
    nullable_source("final_device_used_bytes",
                    "cuda_mem_get_info_device_total_minus_free_device_wide",
                    final_snapshot_available);
    out << ',';
    nullable_source("serialized_plan_bytes", "bundle_engine_section_sizes", true);
    out << ',';
    nullable_source("resident_weight_bytes",
                    "tensorrt_total_weights_size_weight_streaming_disabled",
                    resident_weight_bytes_available);
    out << ',';
    nullable_source("resident_weight_copy_count", "deduplicated_tensorrt_engine_identity",
                    resident_weight_copy_count_available);
    out << ',';
    nullable_source("engine_weight_bytes", "tensorrt_engine_stat_total_weights_size",
                    engine_weight_bytes_available);
    out << ',';
    nullable_source("context_device_memory_bytes", "tensorrt_update_device_memory_size_for_shapes",
                    true);
    out << ',';
    nullable_source("kv_budget_bytes", "runtime_kv_policy_after_reserves_and_overhead", true);
    out << ',';
    nullable_source("external_device_output_bytes",
                    "backend_outputs_plus_runtime_exact_sq_staging_ledger",
                    external_device_output_bytes_available);
    out << ',';
    nullable_source("host_staging_bytes", "backend_host_staging_allocation_ledger",
                    host_staging_bytes_available);
    out << ',';
    nullable_source("graph_private_device_bytes", "cuda_graph_disabled_at_receipt",
                    graph_private_device_bytes_available);
    out << ',';
    nullable_source("kv_reserved_bytes", "runtime_kv_allocator_ledger", true);
    out << ',';
    nullable_source("kv_metadata_bytes", "contiguous_v1_no_device_metadata",
                    kv_metadata_bytes_available);
    out << ',';
    nullable_source("peak_device_bytes",
                    "cuda_mem_get_info_pre_load_free_minus_sampled_free_device_wide_high_water",
                    peak_device_bytes_available);
    out << "},\"peak_device_bytes_unavailable_reason\":";
    if (peak_device_bytes_available) {
        out << "null";
    } else {
        out << "\"" << peak_device_bytes_unavailable_reason << "\"";
    }
    out << '}';
    return out.str();
}

std::shared_ptr<IRuntimeDeviceAllocator> make_cuda_runtime_device_allocator() {
    return std::make_shared<CudaRuntimeDeviceAllocator>();
}

} // namespace trtmc
