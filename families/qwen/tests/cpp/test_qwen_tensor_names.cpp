/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/tensor_names.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        ++g_failures;
    }
}

static void test_expand_simple_i() {
    check(trtmc::qwen_expand_layer_name("cache_k_{i}", 0) == "cache_k_0", "k_{i}_0");
    check(trtmc::qwen_expand_layer_name("cache_k_{i}", 5) == "cache_k_5", "k_{i}_5");
    check(trtmc::qwen_expand_layer_name("cache_k_{i}", 27) == "cache_k_27", "k_{i}_27");
}

static void test_expand_2i() {
    check(trtmc::qwen_expand_layer_name("cache_kv_{2i}", 0) == "cache_kv_0", "kv_{2i}_0");
    check(trtmc::qwen_expand_layer_name("cache_kv_{2i}", 3) == "cache_kv_6", "kv_{2i}_3");
}

static void test_expand_2i_plus_1() {
    check(trtmc::qwen_expand_layer_name("cache_kv_{2i+1}", 0) == "cache_kv_1", "kv_{2i+1}_0");
    check(trtmc::qwen_expand_layer_name("cache_kv_{2i+1}", 3) == "cache_kv_7", "kv_{2i+1}_3");
}

static void test_expand_2i_plus_2() {
    check(trtmc::qwen_expand_layer_name("output{2i+2}", 0) == "output2", "out_{2i+2}_0");
    check(trtmc::qwen_expand_layer_name("output{2i+2}", 4) == "output10", "out_{2i+2}_4");
}

static void test_expand_mixed() {
    check(trtmc::qwen_expand_layer_name("output{2i+1}", 0) == "output1", "out_{2i+1}_0");
    check(trtmc::qwen_expand_layer_name("output{2i+2}", 0) == "output2", "out_{2i+2}_0");
}

static void test_expand_literal() {
    check(trtmc::qwen_expand_layer_name("my_tensor", 5) == "my_tensor", "literal_passthrough");
    check(trtmc::qwen_expand_layer_name("", 0) == "", "empty_pattern");
}

static void test_layer_tensor_name() {
    check(trtmc::qwen_layer_tensor_name("cache_k", 0) == "cache_k_0", "cache_k_0");
    check(trtmc::qwen_layer_tensor_name("present_v", 23) == "present_v_23", "present_v_23");
    check(trtmc::qwen_layer_tensor_name("cache_v", -3) == "cache_v_-3", "negative_layer");
}

int main() {
    test_expand_simple_i();
    test_expand_2i();
    test_expand_2i_plus_1();
    test_expand_2i_plus_2();
    test_expand_mixed();
    test_expand_literal();
    test_layer_tensor_name();

    if (g_failures == 0) {
        std::fprintf(stderr, "All qwen tensor name tests passed.\n");
    } else {
        std::fprintf(stderr, "%d qwen tensor name test(s) FAILED.\n", g_failures);
    }
    return g_failures;
}
