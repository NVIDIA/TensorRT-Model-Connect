/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_memory_plan.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iomanip>
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
    if (fraction == 1.0)
        return bytes;

    // Compute floor(bytes * fraction) from the exact rational represented by
    // the binary64 input. A floating-point multiply rounds before floor() and
    // can therefore over-allocate the user's policy by a few bytes.
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(fraction));
    std::memcpy(&bits, &fraction, sizeof(bits));
    const auto exponent_bits = static_cast<std::uint32_t>((bits >> 52U) & 0x7ffU);
    const auto fraction_bits = bits & ((std::uint64_t{1} << 52U) - 1U);
    std::uint64_t significand = fraction_bits;
    int32_t binary_exponent = -1074;
    if (exponent_bits != 0) {
        significand |= std::uint64_t{1} << 52U;
        binary_exponent = static_cast<int32_t>(exponent_bits) - 1023 - 52;
    }

    __extension__ typedef unsigned __int128 Wide;
    const auto product = static_cast<Wide>(bytes) * static_cast<Wide>(significand);
    if (binary_exponent >= 0) {
        const auto shift = static_cast<std::uint32_t>(binary_exponent);
        if (shift >= 128U ||
            product > (static_cast<Wide>(std::numeric_limits<std::uint64_t>::max()) >> shift)) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return static_cast<std::uint64_t>(product << shift);
    }
    const auto shift = static_cast<std::uint32_t>(-binary_exponent);
    return shift >= 128U ? 0 : static_cast<std::uint64_t>(product >> shift);
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
    if (request.capacity_decision_upper_bound_tokens != 0) {
        if (request.capacity_decision_free_bytes == 0) {
            throw std::invalid_argument(
                "A capacity-decision upper bound requires an authoritative free-memory snapshot");
        }
        if (request.capacity_decision_upper_bound_tokens > semantic_limit(request)) {
            throw std::invalid_argument(
                "The capacity-decision upper bound exceeds the semantic context limit");
        }
    }

    if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
        if (request.policy.bytes == 0)
            throw std::invalid_argument("Explicit KV byte policy must be positive");
    } else if (!(std::isfinite(request.policy.fraction) && request.policy.fraction > 0.0 &&
                 request.policy.fraction <= 1.0)) {
        throw std::invalid_argument("KV cache memory fraction must be in (0, 1]");
    }
}

struct FractionSolveResult {
    std::uint64_t capacity{0};
    RuntimeMemoryOverhead overhead;
    std::vector<std::uint64_t> profiles;
    std::uint32_t iterations{0};
};

std::vector<std::uint64_t> lower_bucket_boundaries(const RuntimeMemoryPlanRequest& request,
                                                   std::uint64_t upper_bound) {
    std::vector<std::uint64_t> boundaries;
    boundaries.reserve(request.active_kv_profile_limits.size() + 1);
    const auto useful_limit = semantic_limit(request);
    for (const auto limit : request.active_kv_profile_limits) {
        if (limit == 0 || limit > request.model_context_limit) {
            throw std::invalid_argument(
                "Active-KV profile limits must be in [1, model_context_limit]");
        }
        if (limit < upper_bound && limit <= useful_limit)
            boundaries.push_back(limit);
    }
    if (upper_bound > 1)
        boundaries.push_back(1);
    std::sort(boundaries.begin(), boundaries.end(), std::greater<std::uint64_t>());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    return boundaries;
}

void count_solve_iteration(std::uint32_t& iterations) {
    if (iterations != std::numeric_limits<std::uint32_t>::max())
        ++iterations;
}

template <typename AvailableBytes>
FractionSolveResult solve_fraction_capacity(const RuntimeMemoryPlanRequest& request,
                                            std::uint64_t initial_candidate,
                                            const RuntimeOverheadQuery& query_overhead,
                                            AvailableBytes&& available_bytes) {
    if (initial_candidate == 0) {
        throw std::runtime_error("Runtime memory budget is smaller than one KV token row");
    }

    const auto useful_limit = semantic_limit(request);
    const auto boundaries = lower_bucket_boundaries(request, initial_candidate);
    auto candidate = initial_candidate;
    FractionSolveResult result;

    const auto evaluate = [&](std::uint64_t rows) {
        FractionSolveResult evaluation;
        evaluation.capacity = rows;
        evaluation.profiles = enabled_profiles(request, rows);
        evaluation.overhead = query_overhead(rows, evaluation.profiles);
        evaluation.capacity =
            std::min(useful_limit,
                     fraction_rows(request.policy.fraction, available_bytes(evaluation.overhead),
                                   request.kv_bytes_per_token));
        return evaluation;
    };

    for (std::uint32_t bounded_iteration = 0; bounded_iteration < request.max_solve_iterations;
         ++bounded_iteration) {
        auto evaluation = evaluate(candidate);
        count_solve_iteration(result.iterations);
        if (candidate <= evaluation.capacity) {
            evaluation.capacity = candidate;
            evaluation.iterations = result.iterations;
            return evaluation;
        }

        const auto next = evaluation.capacity;
        if (next == 0)
            break;
        candidate = next;
    }

    // max_solve_iterations bounds arbitrary fixed-point descent, not safety.
    // Re-query the last candidate and return it only when the exact inequality
    // still holds for O(candidate).
    auto exact = evaluate(candidate);
    count_solve_iteration(result.iterations);
    if (candidate <= exact.capacity) {
        exact.capacity = candidate;
        exact.iterations = result.iterations;
        return exact;
    }

    // Discontinuous tactic/context envelopes can defeat the bounded
    // fixed-point solve one row at a time. Profile limits are the finite lower
    // bucket boundaries that the runtime can safely fall back to.
    for (const auto boundary : boundaries) {
        auto fallback = evaluate(boundary);
        count_solve_iteration(result.iterations);
        if (boundary <= fallback.capacity) {
            fallback.capacity = boundary;
            fallback.iterations = result.iterations;
            return fallback;
        }
    }
    throw std::runtime_error(
        "No runtime KV profile bucket satisfies the exact fraction-memory invariant");
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
    } else if (request.capacity_decision_free_bytes != 0) {
        receipt.kv_budget_bytes = fraction_budget_bytes(
            request.policy.fraction, bytes_after_fixed_reserve(request.capacity_decision_free_bytes,
                                                               request.safety_reserve_bytes));
    } else {
        const auto safe_free =
            bytes_after_fixed_reserve(request.post_load_free_bytes, request.safety_reserve_bytes);
        const auto overhead_bytes = plan.overhead.device_bytes();
        receipt.kv_budget_bytes = fraction_budget_bytes(
            request.policy.fraction, overhead_bytes < safe_free ? safe_free - overhead_bytes : 0);
    }
    receipt.context_device_memory_bytes = plan.overhead.context_device_memory_bytes;
    receipt.ordinary_device_input_bytes = plan.overhead.ordinary_device_input_bytes;
    receipt.ordinary_device_output_bytes = plan.overhead.ordinary_device_output_bytes;
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
    auto total = checked_add(context_device_memory_bytes, ordinary_device_input_bytes,
                             "Runtime device overhead");
    total = checked_add(total, ordinary_device_output_bytes, "Runtime device overhead");
    total = checked_add(total, external_device_output_bytes, "Runtime device overhead");
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

    RuntimeMemoryPlan plan;
    std::uint32_t iterations = 0;
    const bool caller_supplied_capacity_baseline =
        request.capacity_decision_upper_bound_tokens != 0;
    if (caller_supplied_capacity_baseline) {
        const auto upper = request.capacity_decision_upper_bound_tokens;
        if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
            const auto expected =
                std::min(useful_limit, request.policy.bytes / request.kv_bytes_per_token);
            if (expected == 0 || upper != expected) {
                throw std::invalid_argument(
                    "Explicit KV capacity-decision upper bound disagrees with the byte policy");
            }
        } else {
            const auto maximum_without_overhead =
                std::min(useful_limit, fraction_rows(request.policy.fraction, safe_free,
                                                     request.kv_bytes_per_token));
            if (upper > maximum_without_overhead) {
                throw std::invalid_argument(
                    "Capacity-decision upper bound exceeds the post-load policy budget");
            }
        }
        plan.runtime_kv_capacity_tokens = upper;
        plan.allocated_kv_bytes = checked_mul(upper, request.kv_bytes_per_token, "KV cache");
        plan.enabled_profile_limits = enabled_profiles(request, upper);
        plan.overhead = request.capacity_decision_resident_overhead;
    } else {
        std::uint64_t candidate = 0;
        if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
            candidate = std::min(useful_limit, request.policy.bytes / request.kv_bytes_per_token);
        } else {
            candidate = std::min(useful_limit, fraction_rows(request.policy.fraction, safe_free,
                                                             request.kv_bytes_per_token));
        }
        if (candidate == 0) {
            throw std::runtime_error("KV cache memory budget is smaller than one token row (" +
                                     std::to_string(request.kv_bytes_per_token) + " bytes)");
        }

        if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
            plan.enabled_profile_limits = enabled_profiles(request, candidate);
            plan.overhead = query_overhead(candidate, plan.enabled_profile_limits);
            const auto kv_bytes = checked_mul(candidate, request.kv_bytes_per_token, "KV cache");
            const auto required =
                checked_add(kv_bytes, overhead_device_bytes(plan.overhead), "Runtime allocation");
            if (required > safe_free) {
                throw std::runtime_error("Explicit KV policy resolves to " +
                                         std::to_string(kv_bytes) + " KV bytes plus " +
                                         std::to_string(overhead_device_bytes(plan.overhead)) +
                                         " non-KV device bytes, exceeding " +
                                         std::to_string(safe_free) + " safely available bytes");
            }
            plan.runtime_kv_capacity_tokens = candidate;
            plan.allocated_kv_bytes = kv_bytes;
            iterations = 1;
        } else {
            auto solved = solve_fraction_capacity(
                request, candidate, query_overhead, [&](const RuntimeMemoryOverhead& overhead) {
                    const auto overhead_bytes = overhead_device_bytes(overhead);
                    return overhead_bytes < safe_free ? safe_free - overhead_bytes
                                                      : std::uint64_t{0};
                });
            plan.runtime_kv_capacity_tokens = solved.capacity;
            plan.allocated_kv_bytes =
                checked_mul(solved.capacity, request.kv_bytes_per_token, "KV cache");
            plan.enabled_profile_limits = std::move(solved.profiles);
            plan.overhead = solved.overhead;
            iterations = solved.iterations;
        }
    }

    std::uint64_t capacity_resident_overhead_bytes = 0;
    if (request.capacity_decision_free_bytes != 0) {
        const auto final_safe = bytes_after_fixed_reserve(request.capacity_decision_free_bytes,
                                                          request.safety_reserve_bytes);
        capacity_resident_overhead_bytes =
            caller_supplied_capacity_baseline
                ? overhead_device_bytes(request.capacity_decision_resident_overhead)
                : overhead_device_bytes(plan.overhead);
        const auto upper = caller_supplied_capacity_baseline
                               ? request.capacity_decision_upper_bound_tokens
                               : plan.runtime_kv_capacity_tokens;
        // capacity_decision_free_bytes is observed after the tentative
        // context/output overhead has been allocated. Charge only a positive
        // O(final)-O(resident) delta, and never use a later settled snapshot to
        // increase R.
        if (request.policy.kind == RuntimeKvPolicyKind::kBytes) {
            plan.runtime_kv_capacity_tokens = upper;
            plan.allocated_kv_bytes = checked_mul(upper, request.kv_bytes_per_token, "KV cache");
            plan.enabled_profile_limits = enabled_profiles(request, upper);
            plan.overhead = query_overhead(upper, plan.enabled_profile_limits);
            count_solve_iteration(iterations);
            const auto final_overhead_bytes = overhead_device_bytes(plan.overhead);
            const auto positive_delta =
                final_overhead_bytes > capacity_resident_overhead_bytes
                    ? final_overhead_bytes - capacity_resident_overhead_bytes
                    : std::uint64_t{0};
            const auto available_for_kv =
                positive_delta < final_safe ? final_safe - positive_delta : std::uint64_t{0};
            if (plan.allocated_kv_bytes > available_for_kv) {
                throw std::runtime_error(
                    "Available GPU memory changed before the explicit KV allocation: resolved " +
                    std::to_string(plan.allocated_kv_bytes) + " KV bytes, now safely available " +
                    std::to_string(available_for_kv) +
                    " bytes after the final non-KV overhead delta");
            }
        } else {
            const auto final_rows_without_delta =
                fraction_rows(request.policy.fraction, final_safe, request.kv_bytes_per_token);
            auto solved = solve_fraction_capacity(
                request, std::min(upper, final_rows_without_delta), query_overhead,
                [&](const RuntimeMemoryOverhead& overhead) {
                    const auto final_overhead_bytes = overhead_device_bytes(overhead);
                    const auto positive_delta =
                        final_overhead_bytes > capacity_resident_overhead_bytes
                            ? final_overhead_bytes - capacity_resident_overhead_bytes
                            : std::uint64_t{0};
                    return positive_delta < final_safe ? final_safe - positive_delta
                                                       : std::uint64_t{0};
                });
            plan.runtime_kv_capacity_tokens = solved.capacity;
            plan.allocated_kv_bytes =
                checked_mul(solved.capacity, request.kv_bytes_per_token, "KV cache");
            plan.enabled_profile_limits = std::move(solved.profiles);
            plan.overhead = solved.overhead;
            const auto combined_iterations =
                static_cast<std::uint64_t>(iterations) + solved.iterations;
            iterations = combined_iterations > std::numeric_limits<std::uint32_t>::max()
                             ? std::numeric_limits<std::uint32_t>::max()
                             : static_cast<std::uint32_t>(combined_iterations);
        }
    }

    std::uint64_t exact_fraction_policy_available_bytes = 0;
    if (request.policy.kind != RuntimeKvPolicyKind::kBytes) {
        if (request.capacity_decision_free_bytes != 0) {
            const auto final_safe = bytes_after_fixed_reserve(request.capacity_decision_free_bytes,
                                                              request.safety_reserve_bytes);
            const auto final_overhead_bytes = overhead_device_bytes(plan.overhead);
            const auto positive_delta =
                final_overhead_bytes > capacity_resident_overhead_bytes
                    ? final_overhead_bytes - capacity_resident_overhead_bytes
                    : std::uint64_t{0};
            exact_fraction_policy_available_bytes =
                positive_delta < final_safe ? final_safe - positive_delta : std::uint64_t{0};
        } else {
            const auto final_overhead_bytes = overhead_device_bytes(plan.overhead);
            exact_fraction_policy_available_bytes = final_overhead_bytes < safe_free
                                                        ? safe_free - final_overhead_bytes
                                                        : std::uint64_t{0};
        }
        const auto invariant_rows =
            fraction_rows(request.policy.fraction, exact_fraction_policy_available_bytes,
                          request.kv_bytes_per_token);
        if (plan.runtime_kv_capacity_tokens > invariant_rows) {
            throw std::logic_error(
                "Runtime KV solver returned a capacity that violates the exact fraction-memory "
                "invariant");
        }
    }

    plan.receipt = make_receipt(request, plan, iterations);
    if (request.policy.kind != RuntimeKvPolicyKind::kBytes) {
        plan.receipt.kv_budget_bytes =
            fraction_budget_bytes(request.policy.fraction, exact_fraction_policy_available_bytes);
    }
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

    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"receipt_schema_version\":" << receipt_schema_version
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
    nullable_u64("capacity_decision_free_bytes", capacity_decision_free_bytes,
                 capacity_decision_snapshot_available);
    nullable_u64("capacity_decision_total_bytes", capacity_decision_total_bytes,
                 capacity_decision_snapshot_available);
    nullable_u64("capacity_decision_device_used_bytes", capacity_decision_device_used_bytes,
                 capacity_decision_snapshot_available);
    nullable_u64("settled_free_bytes", settled_free_bytes, settled_snapshot_available);
    nullable_u64("settled_total_bytes", settled_total_bytes, settled_snapshot_available);
    nullable_u64("settled_device_used_bytes", settled_device_used_bytes,
                 settled_snapshot_available);
    out << ",\"settled_snapshot_unavailable_reason\":";
    if (settled_snapshot_available) {
        out << "null";
    } else {
        out << "\"" << settled_snapshot_unavailable_reason << "\"";
    }
    nullable_u64("final_free_bytes", final_free_bytes, final_snapshot_available);
    nullable_u64("final_total_bytes", final_total_bytes, final_snapshot_available);
    nullable_u64("final_device_used_bytes", final_device_used_bytes, final_snapshot_available);
    out << ",\"context_device_memory_bytes\":" << context_device_memory_bytes;
    nullable_u64("ordinary_device_input_bytes", ordinary_device_input_bytes,
                 ordinary_device_input_bytes_available);
    nullable_u64("ordinary_device_output_bytes", ordinary_device_output_bytes,
                 ordinary_device_output_bytes_available);
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
    nullable_source("capacity_decision_free_bytes",
                    "cuda_mem_get_info_after_tentative_context_and_output_reservation",
                    capacity_decision_snapshot_available);
    out << ',';
    nullable_source("capacity_decision_total_bytes", "cuda_mem_get_info_device_total",
                    capacity_decision_snapshot_available);
    out << ',';
    nullable_source("capacity_decision_device_used_bytes",
                    "cuda_mem_get_info_device_total_minus_free_device_wide",
                    capacity_decision_snapshot_available);
    out << ',';
    nullable_source("settled_free_bytes",
                    "cuda_mem_get_info_after_final_context_output_and_kv_allocation",
                    settled_snapshot_available);
    out << ',';
    nullable_source("settled_total_bytes", "cuda_mem_get_info_device_total",
                    settled_snapshot_available);
    out << ',';
    nullable_source("settled_device_used_bytes",
                    "cuda_mem_get_info_device_total_minus_free_device_wide",
                    settled_snapshot_available);
    out << ',';
    nullable_source("final_free_bytes", "deprecated_alias_of_capacity_decision_free_bytes",
                    final_snapshot_available);
    out << ',';
    nullable_source("final_total_bytes", "deprecated_alias_of_capacity_decision_total_bytes",
                    final_snapshot_available);
    out << ',';
    nullable_source("final_device_used_bytes",
                    "deprecated_alias_of_capacity_decision_device_used_bytes",
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
    nullable_source("ordinary_device_input_bytes",
                    "backend_concrete_shape_device_input_allocation_ledger",
                    ordinary_device_input_bytes_available);
    out << ',';
    nullable_source("ordinary_device_output_bytes",
                    "backend_concrete_shape_device_output_allocation_ledger",
                    ordinary_device_output_bytes_available);
    out << ',';
    nullable_source("kv_budget_bytes",
                    "runtime_kv_policy_after_reserves_and_positive_non_kv_overhead_delta", true);
    out << ',';
    nullable_source("external_device_output_bytes", "runtime_exact_sq_staging_allocation_ledger",
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
