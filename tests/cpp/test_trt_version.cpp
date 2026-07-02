/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/trt_version.h"

#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

static void check_eq(const std::string& actual, const std::string& expected, const char* name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << name << " actual=\"" << actual << "\" expected=\"" << expected
                  << "\"" << std::endl;
        ++failures;
    }
}

static void test_parse_version() {
    auto v = trtmc::parse_trt_version("10.15.0.6");
    check(v.has_value(), "parse full TRT version");
    check(v->major == 10, "parse major");
    check(v->minor == 15, "parse minor");
    check(v->patch == 0, "parse patch");
    check(v->build == 6, "parse build");
    check_eq(trtmc::trt_abi_string(*v), "10.15", "format ABI string");
    check_eq(trtmc::trt_abi_suffix(*v), "10_15", "format ABI suffix");
    check_eq(trtmc::trt_backend_name_for_abi(*v), "trt_10_15", "format backend name");

    check(!trtmc::parse_trt_version("unknown").has_value(), "unknown version does not parse");
}

static void test_parse_abi_tag() {
    auto dotted = trtmc::parse_trt_abi_tag("10.16");
    check(dotted.has_value(), "parse dotted ABI tag");
    check(dotted->major == 10 && dotted->minor == 16, "dotted ABI values");

    auto backend = trtmc::parse_trt_abi_tag("trt_11_0");
    check(backend.has_value(), "parse backend ABI tag");
    check(backend->major == 11 && backend->minor == 0, "backend ABI values");
}

static void test_matching() {
    auto a = *trtmc::parse_trt_version("10.15.0");
    auto b = *trtmc::parse_trt_version("10.15.2");
    auto c = *trtmc::parse_trt_version("10.16.0");
    check(trtmc::trt_abi_matches(a, b), "patch difference does not change ABI");
    check(!trtmc::trt_abi_matches(a, c), "minor difference changes ABI");
}

static void test_candidates() {
    auto required = trtmc::parse_trt_version("10.15.0");
    auto installed = trtmc::parse_trt_version("10.16.0");

    std::vector<std::string> candidates = trtmc::trt_backend_candidates("trt", required, installed);
    check(candidates.size() == 2, "generic TRT gets versioned plus fallback candidates");
    check_eq(candidates[0], "trt_10_15", "required bundle version wins for candidates");
    check_eq(candidates[1], "trt", "generic TRT fallback candidate");

    candidates = trtmc::trt_backend_candidates("trt", std::nullopt, installed);
    check(candidates.size() == 2, "installed version drives old bundle candidates");
    check_eq(candidates[0], "trt_10_16", "installed version candidate");

    candidates = trtmc::trt_backend_candidates("trt_11_0", required, installed);
    check(candidates.size() == 1, "explicit versioned backend is exact");
    check_eq(candidates[0], "trt_11_0", "explicit versioned backend candidate");

    candidates = trtmc::trt_backend_candidates("trt_rtx", required, installed);
    check(candidates.size() == 1, "TRT-RTX is not standard TRT auto-selected");
    check_eq(candidates[0], "trt_rtx", "TRT-RTX candidate preserved");
}

int main() {
    test_parse_version();
    test_parse_abi_tag();
    test_matching();
    test_candidates();

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}
