/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/domains/text/dynamic_memory/runtime_memory_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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
    request.module_residency_reserve_bytes_by_profile = {0, 0, 0, 0};
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

void test_auto_charges_candidate_module_residency_reserve() {
    auto request = base_request();
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.module_residency_reserve_bytes_by_profile = {10, 20, 100, 200};

    const auto plan =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            RuntimeMemoryOverhead overhead;
            overhead.context_device_memory_bytes = 100;
            return overhead;
        });

    // Initial R=90 is covered by profile 100 and charges 200 reserve bytes,
    // yielding 60 rows. R=60 is covered by profile 64 and charges only 100;
    // the monotonically decreasing solve must not grow again.
    check(plan.runtime_kv_capacity_tokens == 60,
          "auto solve subtracts the candidate-dependent module reserve");
    check(plan.receipt.module_residency_reserve_bytes == 100,
          "receipt records the final candidate module reserve");
    check(plan.receipt.module_residency_reserve_profile_limit == 64,
          "receipt records the lower-bound covering profile");
    check(plan.receipt.kv_budget_bytes == 700,
          "automatic budget excludes safety, module reserve, and overhead");
    check(plan.receipt.receipt_schema_version == 4, "module reserve receipt uses schema 4");
    const auto json = plan.receipt.to_json();
    check(json.find("\"module_residency_reserve_bytes\":100") != std::string::npos,
          "receipt JSON includes the module reserve");
    check(json.find("\"module_residency_reserve_profile_limit\":64") != std::string::npos,
          "receipt JSON includes the reserve covering profile");
    check(json.find("\"module_residency_reserve_bytes\":"
                    "\"plan_bound_profile_calibration\"") != std::string::npos,
          "receipt measurement sources bind reserve bytes to plan calibration");
}

void test_bounded_solve_exhaustion_falls_back_to_exact_lower_bucket() {
    RuntimeMemoryPlanRequest request;
    request.post_load_free_bytes = 1000;
    request.kv_bytes_per_token = 1;
    request.model_context_limit = 1000;
    request.prefill_chunk_limit = 900;
    request.active_kv_profile_limits = {900, 1000};
    request.module_residency_reserve_bytes_by_profile = {0, 0};
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

void test_bucket_fallback_never_regrows_after_crossing_a_profile() {
    RuntimeMemoryPlanRequest request;
    request.post_load_free_bytes = 1000;
    request.kv_bytes_per_token = 1;
    request.model_context_limit = 100;
    request.prefill_chunk_limit = 64;
    request.active_kv_profile_limits = {64, 100};
    request.module_residency_reserve_bytes_by_profile = {0, 0};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.max_solve_iterations = 1;

    std::vector<std::uint64_t> queried_candidates;
    const auto plan = solve_runtime_memory_plan(
        request, [&](std::uint64_t rows, const std::vector<std::uint64_t>&) {
            queried_candidates.push_back(rows);
            RuntimeMemoryOverhead overhead;
            if (rows == 100) {
                overhead.context_device_memory_bytes = 951; // descend to 49
            } else if (rows == 49) {
                overhead.context_device_memory_bytes = 952; // 49 is still unsafe
            } else if (rows == 64) {
                // This crossed profile happens to fit, but selecting it would
                // regrow the already-decreased candidate from 49 to 64.
                overhead.context_device_memory_bytes = 0;
            }
            return overhead;
        });

    check(plan.runtime_kv_capacity_tokens == 1,
          "fallback only considers buckets below the last unsafe candidate");
    check(std::is_sorted(queried_candidates.rbegin(), queried_candidates.rend()),
          "bounded solve never queries a larger candidate after decreasing R");
}

void test_bucket_fallback_includes_below_prefill_chunk_boundary() {
    RuntimeMemoryPlanRequest request;
    request.post_load_free_bytes = 1000;
    request.kv_bytes_per_token = 1;
    request.model_context_limit = 100;
    request.prefill_chunk_limit = 64;
    request.active_kv_profile_limits = {32, 100};
    request.module_residency_reserve_bytes_by_profile = {0, 0};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.max_solve_iterations = 1;

    const auto plan = solve_runtime_memory_plan(
        request, [](std::uint64_t rows, const std::vector<std::uint64_t>&) {
            RuntimeMemoryOverhead overhead;
            if (rows == 100) {
                overhead.context_device_memory_bytes = 930; // descend to 70
            } else if (rows == 70) {
                overhead.context_device_memory_bytes = 931; // still unsafe
            } else if (rows == 63) {
                overhead.context_device_memory_bytes = 0;
            }
            return overhead;
        });

    check(plan.runtime_kv_capacity_tokens == 63,
          "fallback evaluates the finite R=C-1 staging/shape boundary");
}

void test_capacity_decision_larger_overhead_uses_resident_delta_and_bucket_fallback() {
    RuntimeMemoryPlanRequest request;
    request.post_load_free_bytes = 2000;
    request.capacity_decision_free_bytes = 1000;
    request.kv_bytes_per_token = 1;
    request.model_context_limit = 1024;
    request.prefill_chunk_limit = 512;
    request.active_kv_profile_limits = {512, 1024};
    request.module_residency_reserve_bytes_by_profile = {0, 0};
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
    request.module_residency_reserve_bytes_by_profile = {0, 0, 0};
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

    request = base_request();
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kBytes, 0.0, 607};
    const auto rounded =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            return RuntimeMemoryOverhead{};
        });
    check(rounded.runtime_kv_capacity_tokens == 60 && rounded.allocated_kv_bytes == 600,
          "explicit bytes round down by less than one token row");
    check(request.policy.bytes - rounded.allocated_kv_bytes < request.kv_bytes_per_token,
          "explicit byte rounding loses at most B-1 bytes");
}

void test_explicit_bytes_charges_reserve_and_between_bucket_profile() {
    auto request = base_request();
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kBytes, 0.0, 600};
    request.module_residency_reserve_bytes_by_profile = {10, 20, 100, 200};

    const auto plan =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            RuntimeMemoryOverhead overhead;
            overhead.context_device_memory_bytes = 100;
            return overhead;
        });

    check(plan.runtime_kv_capacity_tokens == 60,
          "explicit bytes preserve the requested between-bucket capacity");
    check(plan.receipt.module_residency_reserve_bytes == 100,
          "explicit bytes charge the covering profile reserve");
    check(plan.receipt.module_residency_reserve_profile_limit == 64,
          "between-bucket capacity selects the next active profile");

    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kBytes, 0.0, 800};
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(
                request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
                    return RuntimeMemoryOverhead{};
                });
        },
        "module residency reserve");
}

void test_explicit_bytes_semantic_cap() {
    auto request = base_request();
    request.model_context_limit = 50;
    request.request_context_limit = 40;
    request.active_kv_profile_limits = {16, 32, 50};
    request.module_residency_reserve_bytes_by_profile = {0, 0, 0};
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

void test_synthetic_request_cap_uses_real_covering_profile_reserve() {
    auto request = base_request();
    request.request_context_limit = 40;
    request.module_residency_reserve_bytes_by_profile = {10, 20, 100, 200};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kBytes, 0.0, 800};

    const auto plan =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            return RuntimeMemoryOverhead{};
        });
    check(plan.runtime_kv_capacity_tokens == 40,
          "synthetic U terminal caps the explicit allocation");
    check(plan.receipt.module_residency_reserve_profile_limit == 64,
          "synthetic U terminal selects the smallest real covering profile");
    check(plan.receipt.module_residency_reserve_bytes == 100,
          "synthetic U terminal cannot invent a smaller module reserve");
}

void test_second_snapshot_charges_full_reserve_and_positive_overhead_delta() {
    auto request = base_request();
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.module_residency_reserve_bytes_by_profile = {10, 20, 100, 200};
    request.capacity_decision_free_bytes = 600;
    request.capacity_decision_upper_bound_tokens = 60;
    request.capacity_decision_resident_overhead.context_device_memory_bytes = 40;

    const auto plan = solve_runtime_memory_plan(
        request, [](std::uint64_t rows, const std::vector<std::uint64_t>&) {
            RuntimeMemoryOverhead overhead;
            overhead.context_device_memory_bytes = rows <= 32 ? 190 : 140;
            return overhead;
        });

    // final_safe=500. At R=50, reserve=100 and positive O delta=100,
    // resolving to 30. At R=30, reserve=20 and positive O delta=150,
    // allowing 33; monotonic descent retains R=30.
    check(plan.runtime_kv_capacity_tokens == 30,
          "second snapshot subtracts reserve and positive overhead delta");
    check(plan.receipt.module_residency_reserve_bytes == 20,
          "second snapshot refreshes reserve for the final profile");
    check(plan.receipt.module_residency_reserve_profile_limit == 32,
          "second snapshot records the final covering profile");
    check(plan.receipt.context_device_memory_bytes == 190,
          "second snapshot records final candidate overhead");
    check(plan.receipt.capacity_decision_resident_overhead_bytes == 40,
          "second snapshot records the already-resident tentative overhead");
    check(plan.receipt.final_non_kv_overhead_delta_bytes == 150,
          "second snapshot records the exact positive final overhead delta");
    check(plan.receipt.kv_budget_bytes == 330,
          "second snapshot budget excludes full reserve and positive overhead delta");
    auto serialized_receipt = plan.receipt;
    // The setup layer attaches the authoritative CUDA snapshot after the
    // pure solver returns. Mark that same availability state here so this
    // unit test exercises the non-null serializer path.
    serialized_receipt.capacity_decision_snapshot_available = true;
    const auto receipt_json = serialized_receipt.to_json();
    check(receipt_json.find("\"capacity_decision_resident_overhead_bytes\":40") !=
              std::string::npos,
          "receipt JSON exposes the resident overhead used by F2");
    check(receipt_json.find("\"final_non_kv_overhead_delta_bytes\":150") !=
              std::string::npos,
          "receipt JSON exposes the positive overhead delta used by F2");
}

void test_second_snapshot_does_not_double_charge_resident_overhead() {
    auto request = base_request();
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.module_residency_reserve_bytes_by_profile = {10, 20, 100, 200};
    request.capacity_decision_free_bytes = 600;
    request.capacity_decision_upper_bound_tokens = 60;
    request.capacity_decision_resident_overhead.context_device_memory_bytes = 200;

    const auto plan =
        solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
            RuntimeMemoryOverhead overhead;
            overhead.context_device_memory_bytes = 100;
            return overhead;
        });

    // F2 already reflects the 200 resident bytes. O(final)=100 is smaller,
    // so only safety and Q(40)=100 are charged against F2: (600-100-100)/10.
    check(plan.runtime_kv_capacity_tokens == 40,
          "second snapshot does not subtract resident overhead a second time");
    check(plan.receipt.capacity_decision_resident_overhead_bytes == 200,
          "receipt preserves the resident overhead already reflected by F2");
    check(plan.receipt.final_non_kv_overhead_delta_bytes == 0,
          "smaller final overhead produces no second-stage overhead charge");
}

void test_module_residency_table_and_stale_baseline_fail_closed() {
    auto request = base_request();
    request.module_residency_reserve_bytes_by_profile.clear();
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(
                request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
                    return RuntimeMemoryOverhead{};
                });
        },
        "reserve table must not be empty");

    request = base_request();
    request.module_residency_reserve_bytes_by_profile = {0, 0, 0};
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(
                request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
                    return RuntimeMemoryOverhead{};
                });
        },
        "must align");

    request = base_request();
    request.module_residency_reserve_bytes_by_profile = {0, 100, 50, 200};
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(
                request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
                    return RuntimeMemoryOverhead{};
                });
        },
        "must be nondecreasing");

    request = base_request();
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kAuto, 1.0, 0};
    request.module_residency_reserve_bytes_by_profile = {10, 20, 100, 200};
    request.capacity_decision_free_bytes = 600;
    request.capacity_decision_upper_bound_tokens = 80;
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(
                request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
                    return RuntimeMemoryOverhead{};
                });
        },
        "after module residency");
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

    request.post_load_free_bytes = std::numeric_limits<std::uint64_t>::max();
    request.capacity_decision_free_bytes = request.post_load_free_bytes;
    request.policy.fraction = std::nextafter(1.0, 0.0);
    plan = solve_runtime_memory_plan(request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
        return RuntimeMemoryOverhead{};
    });
    check(plan.receipt.kv_budget_bytes == 18446744073709549567ULL,
          "nextafter-one fraction preserves the exact binary64 floor at uint64 max");
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

void test_overhead_and_explicit_accounting_fail_closed_on_overflow() {
    RuntimeMemoryOverhead overhead;
    overhead.context_device_memory_bytes = std::numeric_limits<std::uint64_t>::max();
    overhead.external_device_output_bytes = 1;
    check_throws([&] { (void)overhead.device_bytes(); }, "overflows uint64");

    auto request = base_request();
    request.post_load_free_bytes = std::numeric_limits<std::uint64_t>::max();
    request.safety_reserve_bytes = 0;
    request.kv_bytes_per_token = 1;
    request.model_context_limit = 1;
    request.prefill_chunk_limit = 1;
    request.active_kv_profile_limits = {1};
    request.module_residency_reserve_bytes_by_profile = {
        std::numeric_limits<std::uint64_t>::max()};
    request.policy = RuntimeKvPolicy{RuntimeKvPolicyKind::kBytes, 0.0, 1};
    check_throws(
        [&] {
            (void)solve_runtime_memory_plan(
                request, [](std::uint64_t, const std::vector<std::uint64_t>&) {
                    return RuntimeMemoryOverhead{};
                });
        },
        "byte count overflows uint64");
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
    test_auto_charges_candidate_module_residency_reserve();
    test_bounded_solve_exhaustion_falls_back_to_exact_lower_bucket();
    test_bucket_fallback_never_regrows_after_crossing_a_profile();
    test_bucket_fallback_includes_below_prefill_chunk_boundary();
    test_capacity_decision_larger_overhead_uses_resident_delta_and_bucket_fallback();
    test_lower_bucket_fallback_respects_request_limit();
    test_explicit_bytes_does_not_silently_shrink();
    test_explicit_bytes_charges_reserve_and_between_bucket_profile();
    test_explicit_bytes_semantic_cap();
    test_synthetic_request_cap_uses_real_covering_profile_reserve();
    test_second_snapshot_charges_full_reserve_and_positive_overhead_delta();
    test_second_snapshot_does_not_double_charge_resident_overhead();
    test_module_residency_table_and_stale_baseline_fail_closed();
    test_final_requery_policy();
    test_capacity_decision_fraction_uses_binary64();
    test_policy_fraction_json_round_trips_binary64();
    test_overhead_and_explicit_accounting_fail_closed_on_overflow();
    test_errors_and_receipt();
    test_sampled_device_high_water_is_not_byte_field_arithmetic();
    if (failures != 0) {
        std::cerr << failures << " runtime-memory-plan checks failed\n";
        return 1;
    }
    std::cout << "runtime-memory-plan checks passed\n";
    return 0;
}
