/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-KV-BUDGET-CPP-01
// Architecture:   ARCH-RT-001
// Unit Design:    UD-RT-KV-01
// Intent:         Runtime KV memory budgeting and sequence admission
// Preconditions:  None (pure CPU arithmetic)
// Postconditions: Runtime budgets are bounded by memory/engine limits and
//                 oversized generation requests fail before inference
// =============================================================================

#include "runtime/domains/text/dynamic_memory/kv_cache_budget.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

template <typename Fn>
void expect_throws(Fn fn, const char* message_fragment, const char* name) {
    try {
        fn();
        std::cerr << "FAIL: " << name << " (no exception thrown)\n";
        ++failures;
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(message_fragment) == std::string::npos) {
            std::cerr << "FAIL: " << name << " (unexpected message: " << error.what() << ")\n";
            ++failures;
        }
    }
}

void test_auto_budget_clamps_to_engine_capability() {
    const trtmc::KvCacheBudget budget = trtmc::resolve_kv_cache_budget({
        /*free_bytes=*/1000,
        /*row_bytes=*/10,
        /*explicit_bytes=*/0,
        /*explicit_fraction=*/0.0,
        /*max_sequence_length=*/0,
        /*engine_max_rows=*/80,
        /*automatic_fraction=*/0.90,
    });

    check(budget.policy == trtmc::KvCacheBudgetPolicy::kAuto, "auto budget policy");
    check(budget.fraction == 0.90, "auto budget fraction");
    check(budget.budget_bytes == 900, "auto budget bytes");
    check(budget.memory_capacity_rows == 90, "auto memory capacity rows");
    check(budget.runtime_rows == 80, "auto rows clamp to engine capability");
    check(budget.allocated_bytes == 800, "auto allocated bytes");
    check(budget.clamped_to_engine_max, "auto reports engine clamp");
}

void test_explicit_fraction_budget() {
    const trtmc::KvCacheBudget budget = trtmc::resolve_kv_cache_budget({
        /*free_bytes=*/1024,
        /*row_bytes=*/8,
        /*explicit_bytes=*/0,
        /*explicit_fraction=*/0.50,
        /*max_sequence_length=*/0,
        /*engine_max_rows=*/256,
    });

    check(budget.policy == trtmc::KvCacheBudgetPolicy::kFraction, "fraction budget policy");
    check(budget.budget_bytes == 512, "fraction budget bytes");
    check(budget.memory_capacity_rows == 64, "fraction memory capacity rows");
    check(budget.runtime_rows == 64, "fraction runtime rows");
    check(budget.allocated_bytes == 512, "fraction allocated bytes");
    check(!budget.clamped_to_engine_max, "fraction does not report engine clamp");
}

void test_explicit_bytes_with_sequence_limit() {
    const trtmc::KvCacheBudget budget = trtmc::resolve_kv_cache_budget({
        /*free_bytes=*/1024,
        /*row_bytes=*/8,
        /*explicit_bytes=*/512,
        /*explicit_fraction=*/0.0,
        /*max_sequence_length=*/32,
        /*engine_max_rows=*/100,
    });

    check(budget.policy == trtmc::KvCacheBudgetPolicy::kBytes, "bytes budget policy");
    check(budget.requested_bytes == 512, "bytes requested budget");
    check(budget.memory_capacity_rows == 64, "bytes memory capacity rows");
    check(budget.runtime_rows == 32, "sequence limit controls runtime rows");
    check(budget.allocated_bytes == 256, "sequence limit controls allocation");
}

void test_runtime_reserve_caps_percentage_and_physical_allocation() {
    const trtmc::KvCacheBudget fraction = trtmc::resolve_kv_cache_budget({
        /*free_bytes=*/1000,
        /*row_bytes=*/10,
        /*explicit_bytes=*/0,
        /*explicit_fraction=*/1.0,
        /*max_sequence_length=*/0,
        /*engine_max_rows=*/100,
        /*automatic_fraction=*/0.90,
        /*minimum_reserve_bytes=*/100,
    });
    check(fraction.reserved_bytes == 100, "fraction safety reserve");
    check(fraction.usable_free_bytes == 900, "fraction usable free bytes");
    check(fraction.budget_bytes == 900, "100 percent is capped below reserve");
    check(fraction.runtime_rows == 90, "reserve caps fraction rows");

    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget({
                /*free_bytes=*/1000,
                /*row_bytes=*/10,
                /*explicit_bytes=*/1000,
                /*explicit_fraction=*/0.0,
                /*max_sequence_length=*/0,
                /*engine_max_rows=*/100,
                /*automatic_fraction=*/0.90,
                /*minimum_reserve_bytes=*/100,
            });
        },
        "safely usable GPU memory", "explicit bytes cannot consume runtime reserve");
}

void test_invalid_budget_requests_fail_early() {
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/0, /*row_bytes=*/8, 0, 0.0, 0, 64});
        },
        "non-zero free GPU memory", "zero free memory rejected");
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/1024, /*row_bytes=*/0, 0, 0.0, 0, 64});
        },
        "row size must be positive", "zero row size rejected");
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/1024, /*row_bytes=*/8, 0, 0.0, 0, 0});
        },
        "engine capability must be positive", "zero engine capability rejected");
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/1024, /*row_bytes=*/8, 512, 0.5, 0, 64});
        },
        "mutually exclusive", "bytes and fraction rejected");
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/1024, /*row_bytes=*/8, 0, 1.01, 0, 64});
        },
        "fraction must be in", "fraction above one rejected");
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/7, /*row_bytes=*/8, 0, 1.0, 0, 64});
        },
        "smaller than one token row", "sub-row budget rejected");
}

void test_requested_sequence_limits_are_enforced() {
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/1024, /*row_bytes=*/8, 0, 1.0, 65, 64});
        },
        "exceeds bundle engine capability", "sequence above engine capability rejected");
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/512, /*row_bytes=*/8, 0, 0.5, 33, 64});
        },
        "does not fit the KV memory budget", "sequence above memory capacity rejected");
    expect_throws(
        [] {
            (void)trtmc::resolve_kv_cache_budget(
                {/*free_bytes=*/128, /*row_bytes=*/8, 256, 0.0, 0, 64});
        },
        "exceeds safely usable GPU memory", "physical allocation above free memory rejected");
}

void test_sequence_admission_is_overflow_safe() {
    trtmc::validate_sequence_admission(/*prompt_tokens=*/8, /*max_new_tokens=*/2,
                                       /*state_max_length=*/10, "test");
    trtmc::validate_sequence_admission(/*prompt_tokens=*/10, /*max_new_tokens=*/0,
                                       /*state_max_length=*/10, "test");

    expect_throws([] { trtmc::validate_sequence_admission(11, 0, 10, "test"); },
                  "exceeds runtime max sequence length", "prompt above capacity rejected");
    expect_throws([] { trtmc::validate_sequence_admission(8, 3, 10, "test"); },
                  "exceeds runtime max sequence length",
                  "generation above remaining capacity rejected");
    expect_throws(
        [] {
            trtmc::validate_sequence_admission(std::numeric_limits<std::size_t>::max(), 1, 10,
                                               "test");
        },
        "exceeds runtime max sequence length", "huge prompt rejected without addition overflow");
    expect_throws([] { trtmc::validate_sequence_admission(1, -1, 10, "test"); },
                  "max_new_tokens must be greater", "negative generation length rejected");
    expect_throws([] { trtmc::validate_sequence_admission(0, 0, 0, "test"); },
                  "capacity must be positive", "non-positive runtime capacity rejected");
}

void test_runtime_memory_admission_preserves_authorities() {
    trtmc::RuntimeSequenceAdmissionContext limits{
        /*model_context_limit=*/100,
        /*runtime_kv_capacity_tokens=*/40,
        /*request_context_limit=*/0,
        /*kv_bytes_per_token=*/16,
        /*kv_budget_bytes=*/704,
        /*kv_reserved_bytes=*/640,
    };
    trtmc::validate_runtime_sequence_admission(30, 10, limits, "test");

    auto semantic = limits;
    semantic.model_context_limit = 35;
    semantic.runtime_kv_capacity_tokens = 35;
    expect_throws([&] { trtmc::validate_runtime_sequence_admission(30, 6, semantic, "test"); },
                  "semantic model context limit exceeded",
                  "model semantic limit has a distinct error");
    expect_throws([&] { trtmc::validate_runtime_sequence_admission(30, 6, semantic, "test"); },
                  "model_context_limit=35", "model semantic error identifies M");

    auto user_policy = limits;
    user_policy.runtime_kv_capacity_tokens = 64;
    user_policy.request_context_limit = 40;
    expect_throws([&] { trtmc::validate_runtime_sequence_admission(32, 9, user_policy, "test"); },
                  "runtime max-sequence policy exceeded",
                  "user max sequence policy has a distinct error");
    expect_throws([&] { trtmc::validate_runtime_sequence_admission(32, 9, user_policy, "test"); },
                  "request_context_limit=40", "user policy error identifies U");

    auto resource = limits;
    resource.runtime_kv_capacity_tokens = 32;
    resource.request_context_limit = 64;
    resource.kv_budget_bytes = 520;
    resource.kv_reserved_bytes = 512;
    expect_throws([&] { trtmc::validate_runtime_sequence_admission(30, 3, resource, "test"); },
                  "runtime KV resource capacity exceeded",
                  "physical runtime capacity has a distinct error");
    expect_throws([&] { trtmc::validate_runtime_sequence_admission(30, 3, resource, "test"); },
                  "required_kv_bytes=528", "resource error converts tokens with B");
    expect_throws([&] { trtmc::validate_runtime_sequence_admission(30, 3, resource, "test"); },
                  "kv_budget_bytes=520", "resource error reports the policy budget");

    trtmc::RuntimeSequenceAdmissionContext overflow{
        /*model_context_limit=*/std::numeric_limits<std::uint64_t>::max(),
        /*runtime_kv_capacity_tokens=*/std::numeric_limits<std::uint64_t>::max() / 2,
        /*request_context_limit=*/0,
        /*kv_bytes_per_token=*/3,
        /*kv_budget_bytes=*/std::numeric_limits<std::uint64_t>::max(),
        /*kv_reserved_bytes=*/std::numeric_limits<std::uint64_t>::max(),
    };
    expect_throws(
        [&] {
            trtmc::validate_runtime_sequence_admission(
                static_cast<std::size_t>(overflow.runtime_kv_capacity_tokens + 1), 0, overflow,
                "test");
        },
        "required_kv_bytes=overflow", "resource byte diagnostic is overflow safe");
}

} // namespace

int main() {
    test_auto_budget_clamps_to_engine_capability();
    test_explicit_fraction_budget();
    test_explicit_bytes_with_sequence_limit();
    test_runtime_reserve_caps_percentage_and_physical_allocation();
    test_invalid_budget_requests_fail_early();
    test_requested_sequence_limits_are_enforced();
    test_sequence_admission_is_overflow_safe();
    test_runtime_memory_admission_preserves_authorities();

    if (failures != 0) {
        std::cerr << failures << " KV cache budget tests failed\n";
        return 1;
    }
    std::cout << "All KV cache budget tests passed\n";
    return 0;
}
