/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-IOMAP-CPP-01
// Architecture:   ARCH-IOMAP-001
// Unit Design:    UD-IOMAP-01
// Intent:         IoMap struct defaults and parse_base_config io_map parsing
//                 from JSON
// Preconditions:  None (pure CPU string logic)
// Postconditions: IoMap defaults and JSON overrides are preserved
// =============================================================================

#include "trtmc/runtime/pipeline_plugin.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", name);
        ++g_failures;
    }
}

// --- IoMap default tests ---

static void test_io_map_defaults() {
    trtmc::IoMap io;
    check(io.token_id == "token_id", "default token_id");
    check(io.position_id == "position_id", "default position_id");
    check(io.attention_mask == "attention_mask", "default attention_mask");
    check(io.logits == "logits", "default logits");
    check(io.cache_k_pattern == "cache_k_{i}", "default cache_k_pattern");
    check(io.cache_v_pattern == "cache_v_{i}", "default cache_v_pattern");
    check(io.present_k_pattern == "present_k_{i}", "default present_k_pattern");
    check(io.present_v_pattern == "present_v_{i}", "default present_v_pattern");
}

// --- BaseConfig io_map field default ---

static void test_base_config_io_map_default() {
    trtmc::BaseConfig cfg;
    check(cfg.io_map.logits == "logits", "base_config io_map.logits default");
    check(cfg.io_map.cache_k_pattern == "cache_k_{i}", "base_config io_map.cache_k default");
}

// --- parse_base_config io_map parsing ---

static void test_parse_io_map_absent() {
    // No io_map in JSON — defaults should remain.
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8
    })";
    auto cfg = trtmc::parse_base_config(config, 128);
    check(cfg.io_map.logits == "logits", "absent io_map: logits default");
    check(cfg.io_map.cache_k_pattern == "cache_k_{i}", "absent io_map: cache_k default");
}

static void test_parse_io_map_present() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "io_map": {
            "logits": "output0",
            "cache_k": "cache_kv_{2i}",
            "cache_v": "cache_kv_{2i+1}",
            "present_k": "output{2i+1}",
            "present_v": "output{2i+2}"
        }
    })";
    auto cfg = trtmc::parse_base_config(config, 128);
    check(cfg.io_map.logits == "output0", "parsed io_map: logits");
    check(cfg.io_map.cache_k_pattern == "cache_kv_{2i}", "parsed io_map: cache_k");
    check(cfg.io_map.cache_v_pattern == "cache_kv_{2i+1}", "parsed io_map: cache_v");
    check(cfg.io_map.present_k_pattern == "output{2i+1}", "parsed io_map: present_k");
    check(cfg.io_map.present_v_pattern == "output{2i+2}", "parsed io_map: present_v");
    // token_id not overridden — should stay as default.
    check(cfg.io_map.token_id == "token_id", "parsed io_map: token_id default preserved");
}

static void test_parse_io_map_partial() {
    // Only override logits, rest stay default.
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "io_map": {
            "logits": "my_logits"
        }
    })";
    auto cfg = trtmc::parse_base_config(config, 64);
    check(cfg.io_map.logits == "my_logits", "partial io_map: logits overridden");
    check(cfg.io_map.cache_k_pattern == "cache_k_{i}", "partial io_map: cache_k default");
    check(cfg.io_map.present_v_pattern == "present_v_{i}", "partial io_map: present_v default");
}

int main() {
    // IoMap struct tests
    test_io_map_defaults();
    test_base_config_io_map_default();

    // parse_base_config io_map tests
    test_parse_io_map_absent();
    test_parse_io_map_present();
    test_parse_io_map_partial();

    if (g_failures == 0)
        fprintf(stderr, "All io_map tests passed.\n");
    else
        fprintf(stderr, "%d io_map test(s) FAILED.\n", g_failures);
    return g_failures;
}
