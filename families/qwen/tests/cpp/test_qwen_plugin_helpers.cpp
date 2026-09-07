/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/plugin_helpers.h"
#include "families/qwen/tests/cpp/native_kv_cache_contract_test.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++failures;
    }
}

template <typename Exception, typename Function>
bool rejects(Function function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void test_row_math() {
    constexpr std::int32_t max_rows = 64;
    constexpr std::uint64_t fp16_row_bytes = 2 * 3 * 4 * 8 * 2;

    check(trtmc::qwen::runtime_cache_rows(0, max_rows, 3, 4, 8, trtmc::DType::kFloat16) == max_rows,
          "zero bytes select bundle capacity");
    check(trtmc::qwen::runtime_cache_rows(fp16_row_bytes * 7, max_rows, 3, 4, 8,
                                          trtmc::DType::kFloat16) == 7,
          "FP16 byte budget selects complete rows");
    check(trtmc::qwen::runtime_cache_rows(fp16_row_bytes * 7 + fp16_row_bytes - 1, max_rows, 3, 4,
                                          8, trtmc::DType::kFloat16) == 7,
          "partial row bytes are not counted");
    check(trtmc::qwen::runtime_cache_rows(fp16_row_bytes * 5, max_rows, 3, 4, 8,
                                          trtmc::DType::kBFloat16) == 5,
          "BF16 uses two-byte cache elements");
    check(trtmc::qwen::runtime_cache_rows(fp16_row_bytes * 2 * 6, max_rows, 3, 4, 8,
                                          trtmc::DType::kFloat32) == 6,
          "FP32 uses four-byte cache elements");
    check(trtmc::qwen::runtime_cache_rows(fp16_row_bytes * 100, max_rows, 3, 4, 8,
                                          trtmc::DType::kFloat16) == max_rows,
          "byte budget clamps to bundle capacity");
    check(rejects<std::invalid_argument>([&] {
              (void)trtmc::qwen::runtime_cache_rows(fp16_row_bytes - 1, max_rows, 3, 4, 8,
                                                    trtmc::DType::kFloat16);
          }),
          "sub-row byte budget is rejected");
    check(rejects<std::overflow_error>([] {
              constexpr auto maximum = std::numeric_limits<std::int32_t>::max();
              (void)trtmc::qwen::runtime_cache_rows(std::numeric_limits<std::uint64_t>::max(), 64,
                                                    maximum, maximum, maximum,
                                                    trtmc::DType::kFloat32);
          }),
          "row byte overflow is rejected");
}

void test_engine_contract() {
    using trtmc::test::NativeKvModuleStub;
    const auto stream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(1));

    NativeKvModuleStub prefill(stream, 2, 64, 4, 8, trtmc::DType::kFloat16, false, nullptr, 4, 16,
                               {}, true, true);
    NativeKvModuleStub decode(stream, 2, 64, 4, 8, trtmc::DType::kFloat16, false, nullptr, 4, 16,
                              {}, true, true);
    trtmc::qwen::validate_kv_row_contract(prefill, true, 2, 32, 64, "prefill");
    trtmc::qwen::validate_kv_row_contract(decode, true, 2, 32, 64, "decode");

    NativeKvModuleStub mixed_prefill(stream, 2, 64, 4, 8, trtmc::DType::kFloat16, false, nullptr, 4,
                                     16, {}, true, true);
    mixed_prefill.set_dynamic("cache_v_1", false);
    check(rejects<std::runtime_error>([&] {
              trtmc::qwen::validate_kv_row_contract(mixed_prefill, true, 2, 32, 64, "prefill");
          }),
          "prefill rejects mixed dynamic and static cache inputs");

    NativeKvModuleStub mixed_decode(stream, 2, 64, 4, 8, trtmc::DType::kFloat16, false, nullptr, 4,
                                    16, {}, true, true);
    mixed_decode.set_dynamic("cache_k_1", false);
    check(rejects<std::runtime_error>([&] {
              trtmc::qwen::validate_kv_row_contract(mixed_decode, true, 2, 32, 64, "decode");
          }),
          "decode rejects mixed dynamic and static cache inputs");

    NativeKvModuleStub wrong_rank(stream, 2, 64, 4, 8, trtmc::DType::kFloat16, false, nullptr, 4,
                                  16, {}, true, true);
    wrong_rank.set_tensor("cache_k_0", {64, 32, 1}, trtmc::DType::kFloat16);
    check(rejects<std::runtime_error>([&] {
              trtmc::qwen::validate_kv_row_contract(wrong_rank, true, 2, 32, 64, "prefill");
          }),
          "dynamic cache rejects a non-rank-2 input");

    NativeKvModuleStub wrong_width(stream, 2, 64, 4, 8, trtmc::DType::kFloat16, false, nullptr, 4,
                                   16, {}, true, true);
    wrong_width.set_tensor("cache_v_0", {64, 31}, trtmc::DType::kFloat16);
    check(rejects<std::runtime_error>([&] {
              trtmc::qwen::validate_kv_row_contract(wrong_width, true, 2, 32, 64, "decode");
          }),
          "dynamic cache rejects the wrong KV width");

    NativeKvModuleStub static_mask(stream, 2, 64, 4, 8, trtmc::DType::kFloat16, false, nullptr, 4,
                                   16, {}, true, false);
    check(rejects<std::runtime_error>([&] {
              trtmc::qwen::validate_kv_row_contract(static_mask, true, 2, 32, 64, "prefill");
          }),
          "dynamic cache rejects a static attention mask");

    NativeKvModuleStub wrong_ceiling(stream, 2, 63, 4, 8, trtmc::DType::kFloat16, false, nullptr, 4,
                                     16, {}, true, true);
    check(rejects<std::runtime_error>([&] {
              trtmc::qwen::validate_kv_row_contract(wrong_ceiling, true, 2, 32, 64, "decode");
          }),
          "dynamic cache rejects a profile below the bundle ceiling");

    NativeKvModuleStub unexpected_dynamic(stream, 2, 64, 4, 8, trtmc::DType::kFloat16, false);
    unexpected_dynamic.set_dynamic("cache_k_1", true);
    check(rejects<std::runtime_error>([&] {
              trtmc::qwen::validate_kv_row_contract(unexpected_dynamic, false, 2, 32, 64, "decode");
          }),
          "static contract rejects one dynamic cache input");
}

} // namespace

int main() {
    test_row_math();
    test_engine_contract();
    return failures;
}
