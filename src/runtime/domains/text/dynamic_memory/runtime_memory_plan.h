/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Shared text-generation runtime-memory planning and allocation contracts.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

enum class RuntimeKvPolicyKind {
    kAuto,
    kFraction,
    kBytes,
};

enum class RuntimeMemoryPeakSampleBoundary {
    kLoadCompletion,
    kRequestCompletion,
};

struct RuntimeKvPolicy {
    RuntimeKvPolicyKind kind{RuntimeKvPolicyKind::kAuto};
    double fraction{0.90};
    std::uint64_t bytes{0};
};

// Non-KV memory that must coexist with the runtime-owned cache for a concrete
// set of actual TensorRT invocation shapes.
struct RuntimeMemoryOverhead {
    std::uint64_t context_device_memory_bytes{0};
    std::uint64_t external_device_output_bytes{0};
    std::uint64_t host_staging_bytes{0};
    std::uint64_t graph_private_device_bytes{0};

    std::uint64_t device_bytes() const;
};

struct RuntimeMemoryPlanRequest {
    std::uint64_t post_load_free_bytes{0};
    // Optional second observation made after context/output allocation. Zero
    // means that the first observation remains authoritative.
    std::uint64_t final_free_bytes{0};
    std::uint64_t safety_reserve_bytes{0};
    std::uint64_t kv_bytes_per_token{0};
    std::uint64_t model_context_limit{0};
    std::uint64_t request_context_limit{0}; // 0 = no additional U cap
    std::uint64_t prefill_chunk_limit{0};
    std::vector<std::uint64_t> active_kv_profile_limits;
    RuntimeKvPolicy policy;
    std::uint32_t max_solve_iterations{8};
};

struct RuntimeMemoryReceipt {
    std::uint32_t receipt_schema_version{2};
    std::uint32_t contract_version{1};
    RuntimeKvPolicyKind policy{RuntimeKvPolicyKind::kAuto};
    double policy_fraction{0.0};
    std::uint64_t requested_kv_bytes{0};
    std::uint64_t post_load_free_bytes{0};
    std::uint64_t safety_reserve_bytes{0};
    std::uint64_t model_context_limit{0};
    std::uint64_t prefill_chunk_limit{0};
    std::uint64_t request_context_limit{0};
    std::uint64_t runtime_kv_capacity_tokens{0};
    std::uint64_t effective_request_limit{0};
    std::uint64_t kv_bytes_per_token{0};
    std::uint64_t kv_budget_bytes{0};

    std::uint64_t pre_load_free_bytes{0};
    std::uint64_t pre_load_total_bytes{0};
    bool pre_load_snapshot_available{false};
    std::uint64_t serialized_plan_bytes{0};
    std::uint64_t resident_weight_bytes{0};
    std::uint32_t resident_weight_copy_count{0};
    std::uint64_t engine_weight_bytes{0};
    bool resident_weight_bytes_available{false};
    bool resident_weight_copy_count_available{false};
    bool engine_weight_bytes_available{false};
    bool weight_streaming_active{false};
    std::uint64_t post_load_total_bytes{0};
    std::uint64_t post_load_device_used_bytes{0};
    bool post_load_total_bytes_available{false};
    std::uint64_t final_free_bytes{0};
    std::uint64_t final_total_bytes{0};
    std::uint64_t final_device_used_bytes{0};
    bool final_snapshot_available{false};
    std::uint64_t context_device_memory_bytes{0};
    std::uint64_t external_device_output_bytes{0};
    std::uint64_t host_staging_bytes{0};
    std::uint64_t graph_private_device_bytes{0};
    std::uint64_t kv_reserved_bytes{0};
    std::uint64_t kv_committed_bytes{0};
    std::uint64_t kv_metadata_bytes{0};
    std::uint64_t peak_device_bytes{0};
    std::uint64_t backend_owned_cache_input_bytes{0};
    std::uint64_t backend_owned_cache_output_bytes{0};
    bool external_device_output_bytes_available{true};
    bool host_staging_bytes_available{true};
    bool graph_private_device_bytes_available{true};
    bool kv_metadata_bytes_available{true};
    bool peak_device_bytes_available{false};
    bool peak_device_sampling_failed{false};
    bool peak_sampled_at_load_completion{false};
    bool peak_sampled_at_request_completion{false};
    std::uint64_t peak_device_sample_count{0};
    std::string peak_device_bytes_unavailable_reason{"no_device_memory_high_water_sample"};

    std::uint64_t kv_allocation_id{0};
    std::uint32_t solve_iterations{0};
    bool capped_by_model{false};
    bool capped_by_request_limit{false};

    void observe_peak_device_memory(std::uint64_t current_free_bytes,
                                    RuntimeMemoryPeakSampleBoundary boundary) noexcept;
    void mark_peak_device_sampling_failed(const char* reason) noexcept;
    std::string to_json() const;
};

struct RuntimeMemoryPlan {
    std::uint64_t runtime_kv_capacity_tokens{0};
    std::uint64_t allocated_kv_bytes{0};
    RuntimeMemoryOverhead overhead;
    std::vector<std::uint64_t> enabled_profile_limits;
    RuntimeMemoryReceipt receipt;
};

using RuntimeOverheadQuery =
    std::function<RuntimeMemoryOverhead(std::uint64_t runtime_kv_capacity_tokens,
                                        const std::vector<std::uint64_t>& enabled_profile_limits)>;

// Resolve a monotonically decreasing cache capacity. The callback must query
// TensorRT with the actual largest enabled Sq/T shapes for the candidate.
RuntimeMemoryPlan solve_runtime_memory_plan(const RuntimeMemoryPlanRequest& request,
                                            const RuntimeOverheadQuery& query_overhead);

// Internal allocator seam used by the native runtime and red-zone tests. The
// first public dynamic-memory release still owns the allocation on behalf of
// the user; this interface deliberately does not expose TensorRT bindings.
struct RuntimeDeviceAllocation {
    void* pointer{nullptr};
    std::uint64_t bytes{0};
    std::uint32_t device{0};
    std::uint64_t alignment{0};
    std::shared_ptr<void> owner;

    bool valid() const { return pointer != nullptr && bytes != 0 && owner != nullptr; }
};

class IRuntimeDeviceAllocator {
  public:
    virtual ~IRuntimeDeviceAllocator() = default;
    virtual RuntimeDeviceAllocation allocate(std::uint64_t bytes, std::uint64_t alignment,
                                             std::uint32_t device, void* stream) = 0;
};

std::shared_ptr<IRuntimeDeviceAllocator> make_cuda_runtime_device_allocator();

const char* runtime_kv_policy_name(RuntimeKvPolicyKind policy);

} // namespace trtmc
