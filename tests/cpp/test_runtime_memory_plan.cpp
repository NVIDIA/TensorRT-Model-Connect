/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_memory_plan.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trtmc;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Fn>
void check_throws(Fn&& fn, const std::string& needle) {
    try {
        fn();
        check(false, "expected exception containing " + needle);
    } catch (const std::exception& error) {
        check(std::string(error.what()).find(needle) != std::string::npos,
              "exception contains '" + needle + "': " + error.what());
    }
}

RuntimeMemoryPlanRequest base_request() {
    RuntimeMemoryPlanRequest request;
    request.post_load_free_bytes = 1000;
    request.safety_reserve_bytes = 100;
    request.kv_bytes_per_token = 10;
    request.model_context_limit = 100;
    request.prefill_chunk_limit = 16;
    request.active_kv_profile_limits = {16, 32, 64, 100};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 0.90, 0};
    return request;
}

void test_auto_bounded_solve() {
    auto request = base_request();
    int queries = 0;
    auto plan = solve_runtime_memory_plan(
        request, [&](std::uint64_t rows, const std::vector<std::uint64_t>&) {
            ++queries;
            RuntimeMemoryOverhead overhead;
            overhead.context_device_memory_bytes = rows <= 64 ? 100 : 200;
            return overhead;
        });
    // r0=81; O=200 -> floor(.9*700/10)=63, then O=100 -> 72 but the
    // solve is decreasing, so it remains 63.
    check(plan.runtime_kv_capacity_tokens == 63, "auto solve decreases monotonically");
    check(plan.allocated_kv_bytes == 630, "auto allocation is R*B");
    check(plan.receipt.kv_budget_bytes == 720,
          "receipt preserves the converged automatic KV budget");
    check(plan.receipt.context_device_memory_bytes == 100,
          "receipt uses converged actual-shape overhead");
    check(queries >= 2, "auto solve queries more than once across discontinuity");
}

void test_bounded_solve_exhaustion_falls_back_to_exact_lower_bucket() {
    RuntimeMemoryPlanRequest request;
    request.post_load_free_bytes = 1000;
    request.kv_bytes_per_token = 1;
    request.model_context_limit = 1000;
    request.prefill_chunk_limit = 900;
    request.active_kv_profile_limits = {900, 1000};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.max_solve_iterations = 8;

    const auto overhead_for = [](std::uint64_t rows) {
        RuntimeMemoryOverhead overhead;
        overhead.context_device_memory_bytes = rows > 900 ? 1001 - rows : 0;
        return overhead;
    };
    const auto plan = solve_runtime_memory_plan(
        request,
        [&](std::uint64_t rows, const std::vector<std::uint64_t>&) { return overhead_for(rows); });

    check(plan.runtime_kv_capacity_tokens == 900,
          "iteration exhaustion falls back to the largest exact lower bucket");
    const auto final_overhead = overhead_for(plan.runtime_kv_capacity_tokens).device_bytes();
    const auto exact_rows = final_overhead < request.post_load_free_bytes
                                ? request.post_load_free_bytes - final_overhead
                                : std::uint64_t{0};
    check(plan.runtime_kv_capacity_tokens <= exact_rows,
          "bounded solve result satisfies the exact final memory invariant");
}

void test_capacity_decision_larger_overhead_uses_resident_delta_and_bucket_fallback() {
    RuntimeMemoryPlanRequest request;
    request.post_load_free_bytes = 2000;
    request.capacity_decision_free_bytes = 1000;
    request.kv_bytes_per_token = 1;
    request.model_context_limit = 1024;
    request.prefill_chunk_limit = 512;
    request.active_kv_profile_limits = {512, 1024};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.max_solve_iterations = 8;

    const auto overhead_for = [](std::uint64_t rows) {
        RuntimeMemoryOverhead overhead;
        if (rows == 1024) {
            overhead.context_device_memory_bytes = 64;
        } else if (rows <= 512) {
            overhead.context_device_memory_bytes = 32;
        } else {
            // O(1000)=96, then each 32-row decrease grows O by 32.
            // The bounded descent must not return an unchecked intermediate.
            overhead.context_device_memory_bytes = 1096 - rows;
        }
        return overhead;
    };
    const auto plan = solve_runtime_memory_plan(
        request,
        [&](std::uint64_t rows, const std::vector<std::uint64_t>&) { return overhead_for(rows); });

    check(plan.runtime_kv_capacity_tokens == 512,
          "F2 solve falls back to the exact lower profile bucket");
    const auto resident_overhead = overhead_for(1024).device_bytes();
    const auto final_overhead = overhead_for(plan.runtime_kv_capacity_tokens).device_bytes();
    const auto positive_delta =
        final_overhead > resident_overhead ? final_overhead - resident_overhead : 0;
    const auto exact_rows = positive_delta < request.capacity_decision_free_bytes
                                ? request.capacity_decision_free_bytes - positive_delta
                                : std::uint64_t{0};
    check(plan.runtime_kv_capacity_tokens <= exact_rows,
          "F2 result satisfies the exact resident-overhead delta invariant");
    check(plan.receipt.context_device_memory_bytes == 32,
          "receipt records the exact final lower-bucket overhead");
}

void test_lower_bucket_fallback_respects_request_limit() {
    RuntimeMemoryPlanRequest request;
    request.post_load_free_bytes = 750;
    request.kv_bytes_per_token = 1;
    request.model_context_limit = 1024;
    request.request_context_limit = 750;
    request.prefill_chunk_limit = 512;
    request.active_kv_profile_limits = {256, 512, 1024};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.max_solve_iterations = 2;

    const auto plan = solve_runtime_memory_plan(
        request, [](std::uint64_t rows, const std::vector<std::uint64_t>&) {
            RuntimeMemoryOverhead overhead;
            overhead.context_device_memory_bytes = rows > 512 ? 751 - rows : 0;
            return overhead;
        });

    check(plan.runtime_kv_capacity_tokens == 512,
          "fallback selects the largest safe active bucket below U");
    check(plan.enabled_profile_limits == std::vector<std::uint64_t>({256, 512}),
          "profiles above U are not enabled by the fallback");
}

void test_explicit_bytes_does_not_silently_shrink() {
    auto request = base_request();
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kBytes, 0.0, 800};
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(request,
                                            [](std::uint64_t, const std::vector<std::uint64_t>&) {
                                                RuntimeMemoryOverhead overhead;
                                                overhead.context_device_memory_bytes = 200;
                                                return overhead;
                                            });
        },
        "Explicit KV policy");
}

void test_explicit_bytes_semantic_cap() {
    auto request = base_request();
    request.model_context_limit = 50;
    request.request_context_limit = 40;
    request.active_kv_profile_limits = {16, 32, 50};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kBytes, 0.0, 800};
    auto plan =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            return RuntimeMemoryOverhead{};
        });
    check(plan.runtime_kv_capacity_tokens == 40, "U caps semantically useful allocation");
    check(plan.allocated_kv_bytes == 400, "large byte request does not reserve unusable rows");
    check(plan.receipt.kv_budget_bytes == 800,
          "receipt preserves explicit bytes independently from reserved bytes");
}

void test_final_requery_policy() {
    auto automatic = base_request();
    automatic.capacity_decision_free_bytes = 700;
    auto auto_plan =
        solve_runtime_memory_plan(automatic, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            RuntimeMemoryOverhead overhead;
            overhead.context_device_memory_bytes = 100;
            return overhead;
        });
    check(auto_plan.runtime_kv_capacity_tokens == 54,
          "auto recomputes from the post-overhead capacity-decision reading");

    auto still_fits_but_fraction_changed = base_request();
    still_fits_but_fraction_changed.capacity_decision_free_bytes = 850;
    auto fraction_plan = solve_runtime_memory_plan(
        still_fits_but_fraction_changed, [](std::uint64_t rows, const std::vector<std::uint64_t>&) {
            RuntimeMemoryOverhead overhead;
            overhead.context_device_memory_bytes = rows > 64 ? 100 : 50;
            return overhead;
        });
    check(fraction_plan.runtime_kv_capacity_tokens == 67,
          "final requery preserves fraction semantics even when the old KV still fits");
    check(fraction_plan.receipt.context_device_memory_bytes == 100,
          "final requery refreshes overhead for the resolved capacity");

    auto explicit_request = base_request();
    explicit_request.capacity_decision_free_bytes = 650;
    explicit_request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kBytes, 0.0, 600};
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(explicit_request,
                                            [](std::uint64_t, const std::vector<std::uint64_t>&) {
                                                RuntimeMemoryOverhead overhead;
                                                overhead.context_device_memory_bytes = 100;
                                                return overhead;
                                            });
        },
        "Available GPU memory changed");
}

void test_capacity_decision_fraction_uses_binary64() {
    auto request = base_request();
    request.post_load_free_bytes = 1234567890123456789ULL;
    request.capacity_decision_free_bytes = 1234567890123456789ULL;
    request.safety_reserve_bytes = 0;
    auto plan =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            return RuntimeMemoryOverhead{};
        });
    check(plan.receipt.kv_budget_bytes == 1111111101111111137ULL,
          "capacity-decision budget floors the exact binary64 ratio");

    request.post_load_free_bytes = (std::uint64_t{1} << 63U) - 1U;
    request.capacity_decision_free_bytes = request.post_load_free_bytes;
    plan = solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
        return RuntimeMemoryOverhead{};
    });
    check(plan.receipt.kv_budget_bytes == 8301034833169298431ULL,
          "binary64 ratio floor never rounds a large budget upward");
}

void test_policy_fraction_json_round_trips_binary64() {
    auto request = base_request();
    request.policy.fraction = 0.12345678901234567;
    const auto plan =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            return RuntimeMemoryOverhead{};
        });
    const auto json = plan.receipt.to_json();
    const std::string marker = "\"policy_fraction\":";
    const auto begin = json.find(marker);
    check(begin != std::string::npos, "receipt JSON contains policy_fraction");
    if (begin == std::string::npos)
        return;
    const auto value_begin = begin + marker.size();
    const auto value_end = json.find(',', value_begin);
    const auto parsed = std::stod(json.substr(value_begin, value_end - value_begin));
    check(parsed == request.policy.fraction,
          "receipt JSON policy_fraction round-trips the original binary64 value");
}

void test_errors_and_receipt() {
    auto request = base_request();
    request.request_context_limit = 101;
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(request,
                                            [](std::uint64_t, const std::vector<std::uint64_t>&) {
                                                return RuntimeMemoryOverhead{};
                                            });
        },
        "exceeds the model context limit");

    request = base_request();
    request.policy.fraction = 0.0;
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(request,
                                            [](std::uint64_t, const std::vector<std::uint64_t>&) {
                                                return RuntimeMemoryOverhead{};
                                            });
        },
        "fraction");

    request = base_request();
    auto plan =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            return RuntimeMemoryOverhead{};
        });
    const auto json = plan.receipt.to_json();
    check(json.find("\"runtime_kv_capacity_tokens\":81") != std::string::npos,
          "receipt JSON includes R");
    check(json.find("\"kv_budget_bytes\":810") != std::string::npos,
          "receipt JSON includes the resolved KV budget");
    check(json.find("\"backend_owned_cache_input_bytes\":0") != std::string::npos,
          "receipt JSON includes backend cache ownership");
}

void test_sampled_device_high_water_is_not_byte_field_arithmetic() {
    RuntimeMemoryReceipt receipt;
    receipt.pre_load_snapshot_available = true;
    receipt.pre_load_free_bytes = 1000;
    receipt.pre_load_total_bytes = 2000;
    receipt.context_device_memory_bytes = 9000;
    receipt.kv_reserved_bytes = 8000;

    receipt.observe_peak_device_memory(700, RuntimeMemoryPeakSampleBoundary::kLoadCompletion);
    receipt.observe_peak_device_memory(650, RuntimeMemoryPeakSampleBoundary::kRequestCompletion);
    check(receipt.peak_device_bytes_available, "sampled peak becomes available");
    check(receipt.peak_device_bytes == 350,
          "sampled peak is pre-load free minus minimum sampled free");
    check(receipt.peak_device_bytes !=
              receipt.context_device_memory_bytes + receipt.kv_reserved_bytes,
          "sampled peak is not synthesized from receipt byte fields");
    check(receipt.peak_device_sample_count == 2, "load and request boundaries are counted");
    const auto json = receipt.to_json();
    check(json.find("\"peak_device_bytes\":350") != std::string::npos,
          "receipt serializes sampled peak");
    check(json.find("\"after_successful_request_completion\"") != std::string::npos,
          "receipt serializes request sample boundary");

    receipt.mark_peak_device_sampling_failed("request_completion_cuda_mem_get_info_failed");
    check(!receipt.peak_device_bytes_available,
          "failed sampling invalidates an incomplete high-water");
    check(receipt.to_json().find("\"peak_device_bytes_unavailable_reason\":"
                                 "\"request_completion_cuda_mem_get_info_failed\"") !=
              std::string::npos,
          "sampling failure remains explicit");
}

} // namespace

int main() {
    test_auto_bounded_solve();
    test_bounded_solve_exhaustion_falls_back_to_exact_lower_bucket();
    test_capacity_decision_larger_overhead_uses_resident_delta_and_bucket_fallback();
    test_lower_bucket_fallback_respects_request_limit();
    test_explicit_bytes_does_not_silently_shrink();
    test_explicit_bytes_semantic_cap();
    test_final_requery_policy();
    test_capacity_decision_fraction_uses_binary64();
    test_policy_fraction_json_round_trips_binary64();
    test_errors_and_receipt();
    test_sampled_device_high_water_is_not_byte_field_arithmetic();
    if (failures != 0) {
        std::cerr << failures << " runtime-memory-plan checks failed\n";
        return 1;
    }
    std::cout << "runtime-memory-plan checks passed\n";
    return 0;
}
