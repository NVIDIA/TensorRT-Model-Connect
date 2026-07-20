/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/text_timing.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_normal_timing_keeps_numeric_output() {
    const auto metrics = trtmc::cli::summarize_text_benchmark({
        {4.0, 10.0, 5},
        {6.0, 30.0, 15},
    });
    check(metrics.timing_available, "normal timing is available");
    check(metrics.prefill_ms == 5.0, "normal prefill mean");
    check(metrics.decode_ms == 20.0, "normal decode mean");
    check(metrics.tokens_per_sec == 500.0, "normal aggregate throughput");
    check(trtmc::cli::format_text_benchmark(metrics) ==
              "[trtmc.benchmark] prefill_ms=5.00 decode_ms=20.00 tokens_per_sec=500.00",
          "normal benchmark format is backward compatible");
    check(trtmc::cli::format_text_timing(4.0, 10.0) ==
              "[trtmc.timing] prefill_ms=4.000000 decode_ms=10.000000 total_ms=14.000000",
          "normal timing format is backward compatible");
}

void test_zero_timing_is_explicitly_unavailable() {
    const auto metrics =
        trtmc::cli::summarize_text_benchmark({trtmc::cli::TextTimingSample{0.0, 0.0, 32}});
    check(!metrics.timing_available, "zero timing is unavailable");
    check(trtmc::cli::format_text_benchmark(metrics) ==
              "[trtmc.benchmark] prefill_ms=unavailable decode_ms=unavailable "
              "tokens_per_sec=unavailable",
          "zero benchmark never emits infinity");
    check(trtmc::cli::format_text_timing(0.0, 0.0) ==
              "[trtmc.timing] prefill_ms=unavailable decode_ms=unavailable "
              "total_ms=unavailable",
          "zero single-run timing is unavailable");
}

void test_non_finite_timing_is_explicitly_unavailable() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (const auto& sample : std::vector<trtmc::cli::TextTimingSample>{
             {nan, 10.0, 4},
             {1.0, nan, 4},
             {infinity, 10.0, 4},
             {1.0, infinity, 4},
             {1.0, -1.0, 4},
         }) {
        const auto metrics = trtmc::cli::summarize_text_benchmark({sample});
        const std::string output = trtmc::cli::format_text_benchmark(metrics);
        check(!metrics.timing_available, "non-finite or negative timing is unavailable");
        check(output.find("unavailable") != std::string::npos,
              "invalid timing renders unavailable");
        check(output.find("inf") == std::string::npos && output.find("nan") == std::string::npos,
              "invalid timing never leaks inf or nan");
    }
}

void test_throughput_uses_actual_generated_tokens() {
    const auto metrics = trtmc::cli::summarize_text_benchmark({
        {2.0, 10.0, 3},
        {2.0, 10.0, 5},
    });
    check(metrics.timing_available, "early-EOS timing is available");
    check(metrics.tokens_per_sec == 400.0,
          "throughput uses eight actual tokens rather than requested token budget");
}

} // namespace

int main() {
    test_normal_timing_keeps_numeric_output();
    test_zero_timing_is_explicitly_unavailable();
    test_non_finite_timing_is_explicitly_unavailable();
    test_throughput_uses_actual_generated_tokens();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All CLI text timing tests passed\n";
    return EXIT_SUCCESS;
}
