// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-IOMAP-CPP-01
// Architecture:   ARCH-IOMAP-001
// Unit Design:    UD-IOMAP-01
// Intent:         IoMap struct defaults, expand_layer_name pattern expansion,
//                 parse_base_config io_map parsing from JSON
// Preconditions:  None (pure CPU string logic)
// Postconditions: Pattern tokens are expanded correctly for all layer indices
// =============================================================================

#include "runtime/core/trt_engine_lifecycle.h"
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

// --- expand_layer_name tests ---

static void test_expand_simple_i() {
    check(trtmc::expand_layer_name("cache_k_{i}", 0) == "cache_k_0", "k_{i}_0");
    check(trtmc::expand_layer_name("cache_k_{i}", 5) == "cache_k_5", "k_{i}_5");
    check(trtmc::expand_layer_name("cache_k_{i}", 27) == "cache_k_27", "k_{i}_27");
}

static void test_expand_2i() {
    check(trtmc::expand_layer_name("cache_kv_{2i}", 0) == "cache_kv_0", "kv_{2i}_0");
    check(trtmc::expand_layer_name("cache_kv_{2i}", 3) == "cache_kv_6", "kv_{2i}_3");
}

static void test_expand_2i_plus_1() {
    check(trtmc::expand_layer_name("cache_kv_{2i+1}", 0) == "cache_kv_1", "kv_{2i+1}_0");
    check(trtmc::expand_layer_name("cache_kv_{2i+1}", 3) == "cache_kv_7", "kv_{2i+1}_3");
}

static void test_expand_2i_plus_2() {
    check(trtmc::expand_layer_name("output{2i+2}", 0) == "output2", "out_{2i+2}_0");
    check(trtmc::expand_layer_name("output{2i+2}", 4) == "output10", "out_{2i+2}_4");
}

static void test_expand_mixed() {
    // Pattern with both {2i+1} and {2i+2} — each replaced independently.
    check(trtmc::expand_layer_name("output{2i+1}", 0) == "output1", "out_{2i+1}_0");
    check(trtmc::expand_layer_name("output{2i+2}", 0) == "output2", "out_{2i+2}_0");
}

static void test_expand_literal() {
    // No tokens — should return the pattern unchanged.
    check(trtmc::expand_layer_name("my_tensor", 5) == "my_tensor", "literal_passthrough");
    check(trtmc::expand_layer_name("", 0) == "", "empty_pattern");
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
    // expand_layer_name tests
    test_expand_simple_i();
    test_expand_2i();
    test_expand_2i_plus_1();
    test_expand_2i_plus_2();
    test_expand_mixed();
    test_expand_literal();

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
