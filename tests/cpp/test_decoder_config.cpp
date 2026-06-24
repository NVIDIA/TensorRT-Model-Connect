// =============================================================================
// test_decoder_config.cpp -- Unit tests for decoder config parsing
// =============================================================================
//
// Purpose:
//   Validates BaseConfig parsing for decoder strategies across the C++ runtime:
//   - BaseConfig parsing with runtime_strategy="decoder_kv_cache"
//   - GQA vs MHA attention_size computation
//   - Tokenizer config fields
//   - Cache length override and cap
//
// Dependencies:
//   - trtmc/runtime/pipeline_plugin.h (parse_base_config, BaseConfig)
//
// Environment:
//   CPU-only for config parsing tests.
//
// Note: Engine tensor naming and logits output naming are now auto-detected
// at runtime by KvCache::bind_to() and TextGenerationPipeline's constructor,
// respectively. No explicit naming scheme selection is needed.
// =============================================================================

#include "trtmc/runtime/pipeline_plugin.h"

#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// =============================================================================
// Config parsing tests -- decoder strategy
// =============================================================================

// -----------------------------------------------------------------------------
// Intention: Verify that runtime_strategy="decoder_kv_cache" is parsed correctly
//            and all standard decoder fields are populated.
// Setup:     JSON config mimicking a Qwen3-0.6B bundle.
// Mechanism: parse_base_config, assert strategy and architecture fields.
// -----------------------------------------------------------------------------
static void test_decoder_config_basic() {
    const std::string config = R"({
        "vocab_size": 151936,
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "runtime_strategy": "decoder_kv_cache",
        "max_position_embeddings": 32768
    })";

    const auto cfg = trtmc::parse_base_config(config, 256);
    check(cfg.runtime_strategy == "decoder_kv_cache", "decoder: runtime_strategy parsed");
    check(cfg.vocab_size == 151936, "decoder: vocab_size");
    check(cfg.hidden_size == 1024, "decoder: hidden_size");
    check(cfg.num_layers == 28, "decoder: num_layers");
    check(cfg.num_heads == 16, "decoder: num_heads");
    check(cfg.num_kv_heads == 2, "decoder: num_kv_heads");
    check(cfg.head_dim == 64, "decoder: head_dim");
    check(cfg.attention_size == 16 * 64, "decoder: attention_size = 1024");
    check(cfg.max_cache_length == 256, "decoder: max_cache_length (override)");
}

// -----------------------------------------------------------------------------
// Intention: Verify that decoder with GQA (num_kv_heads < num_heads)
//            computes attention_size from num_attention_heads * head_dim.
//            The raw KV cache width is inferred from engine tensors and uses
//            num_key_value_heads * head_dim.
// Setup:     Config with num_attention_heads=16, num_key_value_heads=2 (GQA 8:1).
// Mechanism: Assert attention_size = 16 * 64 = 1024.
// -----------------------------------------------------------------------------
static void test_decoder_gqa_attention_size() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "runtime_strategy": "decoder_kv_cache"
    })";

    const auto cfg = trtmc::parse_base_config(config, 128);
    // attention_size remains the query projection width.
    check(cfg.attention_size == 16 * 64,
          "decoder GQA: attention_size = num_heads * head_dim = 1024");
}

// -----------------------------------------------------------------------------
// Intention: Verify that decoder with MHA (num_kv_heads == num_heads)
//            works correctly -- no GQA expansion needed.
// Setup:     Config with num_attention_heads=16, num_key_value_heads=16 (MHA).
// Mechanism: Assert attention_size = 16 * 128 = 2048.
// -----------------------------------------------------------------------------
static void test_decoder_mha() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "runtime_strategy": "decoder_kv_cache"
    })";

    const auto cfg = trtmc::parse_base_config(config, 512);
    check(cfg.num_heads == cfg.num_kv_heads, "decoder MHA: heads == kv_heads");
    check(cfg.attention_size == 16 * 128, "decoder MHA: attention_size = 2048");
}

// -----------------------------------------------------------------------------
// Intention: Verify tokenizer_add_special_tokens field is parsed from bundle
//            config. The builder detects this at build time.
// Setup:     Config with tokenizer_add_special_tokens=1.
// Mechanism: Assert the field is parsed correctly.
// -----------------------------------------------------------------------------
static void test_decoder_tokenizer_config() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "runtime_strategy": "decoder_kv_cache",
        "tokenizer_add_special_tokens": 1,
        "bos_token_id": 151643,
        "eos_token_id": [151645, 151643]
    })";

    const auto cfg = trtmc::parse_base_config(config, 128);
    check(cfg.tokenizer_add_special_tokens == true, "decoder: tokenizer_add_special_tokens = true");
    check(cfg.tokenizer_add_special_tokens_present == true,
          "decoder: tokenizer_add_special_tokens_present = true");
    check(cfg.id_bos == 151643, "decoder: bos_token_id");
    check(cfg.id_eos == 151645, "decoder: eos_token_id from array");
}

// -----------------------------------------------------------------------------
// Intention: Verify that max_cache_length respects the override, not the
//            config's max_position_embeddings, since bundles have
//            max_cache_length baked in at build time.
// Setup:     Config with max_position_embeddings=32768, override=256.
// Mechanism: Assert max_cache_length = 256.
// -----------------------------------------------------------------------------
static void test_decoder_cache_length_override() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_position_embeddings": 32768,
        "runtime_strategy": "decoder_kv_cache"
    })";

    // Override = 256 (from bundle config's max_cache_length)
    const auto cfg = trtmc::parse_base_config(config, 256);
    check(cfg.max_cache_length == 256, "decoder: cache_length override = 256");
}

// -----------------------------------------------------------------------------
// Intention: Verify that decoder config without max_cache_length
//            override still caps at 4096 (same as other decoder strategies).
// Setup:     Config with max_position_embeddings=131072, no override.
// Mechanism: Assert max_cache_length = 4096 (cap).
// -----------------------------------------------------------------------------
static void test_decoder_cache_length_cap() {
    const std::string config = R"({
        "vocab_size": 32000,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "runtime_strategy": "decoder_kv_cache"
    })";

    const auto cfg = trtmc::parse_base_config(config, -1);
    check(cfg.max_cache_length == 4096, "decoder: cache_length capped at 4096");
}

int main() {
    // Config parsing tests (no TRT needed)
    test_decoder_config_basic();
    test_decoder_gqa_attention_size();
    test_decoder_mha();
    test_decoder_tokenizer_config();
    test_decoder_cache_length_override();
    test_decoder_cache_length_cap();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All decoder config tests passed.\n";
    return 0;
}
