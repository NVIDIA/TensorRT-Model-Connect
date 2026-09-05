/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for runtime plugin helper parsing.
// Focus: tokenizer_add_special_tokens detection from bundle config.

#include "../../native_kv_cache_contract_test.h"
#include "runtime/models/smollm3/plugin_helpers.h"

#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

static trtmc::BundleFile make_bundle_with_config(const std::string& config) {
    trtmc::BundleFile bundle;
    trtmc::BundleSection sec;
    sec.name = "config.json";
    sec.data.assign(config.begin(), config.end());
    bundle.sections.push_back(std::move(sec));
    return bundle;
}

static trtmc::BundleFile make_bundle_with_config_and_tokenizer(const std::string& config,
                                                               const std::string& tokenizer_json) {
    auto bundle = make_bundle_with_config(config);
    trtmc::BundleSection tok;
    tok.name = "tokenizer.json";
    tok.data.assign(tokenizer_json.begin(), tokenizer_json.end());
    bundle.sections.push_back(std::move(tok));
    return bundle;
}

static void check_ids(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected,
                      const char* name) {
    if (actual != expected) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

static void test_missing_field_defaults_true() {
    auto bundle = make_bundle_with_config(R"({"runtime_strategy":"smollm3_decoder_kv_cache"})");
    check(trtmc::detect_add_special_tokens(bundle) == true,
          "detect_add_special_tokens: missing field defaults true");
}

static void test_integer_false_parsed() {
    auto bundle = make_bundle_with_config(R"({"tokenizer_add_special_tokens":0})");
    check(trtmc::detect_add_special_tokens(bundle) == false,
          "detect_add_special_tokens: integer 0 parsed as false");
}

static void test_integer_true_parsed() {
    auto bundle = make_bundle_with_config(R"({"tokenizer_add_special_tokens":1})");
    check(trtmc::detect_add_special_tokens(bundle) == true,
          "detect_add_special_tokens: integer 1 parsed as true");
}

static void test_bool_false_parsed() {
    auto bundle = make_bundle_with_config(R"({"tokenizer_add_special_tokens":false})");
    check(trtmc::detect_add_special_tokens(bundle) == false,
          "detect_add_special_tokens: bool false parsed as false");
}

static void test_bool_true_parsed() {
    auto bundle = make_bundle_with_config(R"({"tokenizer_add_special_tokens":true})");
    check(trtmc::detect_add_special_tokens(bundle) == true,
          "detect_add_special_tokens: bool true parsed as true");
}

static void test_decoder_profile_selection_keeps_runtime_ceiling() {
    check_ids(trtmc::select_decoder_profile_rows({256, 131072}, 1000), {256, 131072},
              "decoder profile selection keeps first runtime ceiling");
    check_ids(trtmc::select_decoder_profile_rows({256, 131072}, 256), {256},
              "decoder profile selection stops at exact runtime capacity");
    bool rejected = false;
    try {
        (void)trtmc::select_decoder_profile_rows({256}, 131072);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "decoder profile selection rejects an undersized largest bucket");
}

static void test_dynamic_profile_rows_use_profile_metadata() {
    trtmc::test::NativeKvModuleStub module(nullptr, 1, 131072, 1, 2, trtmc::DType::kFloat16,
                                           /*native=*/false, nullptr, 4, 16,
                                           /*dynamic_legacy_cache=*/true, {131072, 256, 131072},
                                           {131072, 1, 1});

    check(module.tensor_shape("cache_k_0") == std::vector<int64_t>{131072, 2},
          "dynamic module reports a positive active cache shape");
    check(trtmc::cache_input_supports_runtime_rows(module, "cache_k_0"),
          "dynamic input metadata enables runtime rows despite positive active shape");
    check(trtmc::decoder_profile_cache_rows(module, "cache_k_0", 1, 131072) == 256,
          "first dynamic decode profile uses its own row ceiling");
    check(trtmc::decoder_profile_cache_rows(module, "cache_k_0", 2, 131072) == 131072,
          "second dynamic decode profile uses its own row ceiling");

    const auto roles = trtmc::detect_decoder_profile_roles(module, "token_id", "cache_k_0", 131072);
    check(roles.prefill_profile_idx == 0 && roles.prefill_max_length == 131072,
          "role detection keeps the dynamic prefill profile");
    check(roles.decode_profiles.size() == 2 && roles.decode_profiles[0].profile_idx == 1 &&
              roles.decode_profiles[0].kv_rows == 256 &&
              roles.decode_profiles[1].profile_idx == 2 &&
              roles.decode_profiles[1].kv_rows == 131072,
          "role detection preserves per-profile dynamic KV ceilings");
}

static void test_exact_special_frame_overrides_native_unigram_fallback() {
    const std::string tokenizer_json = R"({
      "model": {
        "type": "Unigram",
        "unk_id": 0,
        "vocab": [
          ["<unk>", 0.0],
          ["<s>", 0.0],
          ["</s>", 0.0],
          ["\u2581", -1.0],
          ["h", -1.0],
          ["e", -1.0]
        ]
      },
      "pre_tokenizer": {
        "type": "Metaspace",
        "replacement": "\u2581",
        "add_prefix_space": true
      }
    })";
    auto bundle = make_bundle_with_config_and_tokenizer(
        R"({
          "tokenizer_add_special_tokens": 1,
          "tokenizer_special_prefix_ids": [1],
          "tokenizer_special_suffix_ids": []
        })",
        tokenizer_json);

    auto tokenizer = trtmc::create_tokenizer_from_bundle(bundle);
    check(tokenizer != nullptr, "create native tokenizer with exact special frame");
    check_ids(tokenizer->encode("he"), {1, 3, 4, 5},
              "exact special frame adds BOS without fallback EOS");
}

static void test_exact_special_frame_respects_add_special_false() {
    const std::string tokenizer_json = R"({
      "model": {
        "type": "Unigram",
        "unk_id": 0,
        "vocab": [
          ["<unk>", 0.0],
          ["<s>", 0.0],
          ["</s>", 0.0],
          ["\u2581", -1.0],
          ["h", -1.0],
          ["e", -1.0]
        ]
      },
      "pre_tokenizer": {
        "type": "Metaspace",
        "replacement": "\u2581",
        "add_prefix_space": true
      }
    })";
    auto bundle = make_bundle_with_config_and_tokenizer(
        R"({
          "tokenizer_add_special_tokens": 1,
          "tokenizer_special_prefix_ids": [1],
          "tokenizer_special_suffix_ids": [2]
        })",
        tokenizer_json);

    auto tokenizer = trtmc::try_create_native_tokenizer(bundle, /*add_special_tokens=*/false);
    check(tokenizer != nullptr, "create native tokenizer with special frame disabled");
    check_ids(tokenizer->encode("he"), {3, 4, 5},
              "exact special frame does not override add_special_tokens=false");
}

int main() {
    test_missing_field_defaults_true();
    test_integer_false_parsed();
    test_integer_true_parsed();
    test_bool_false_parsed();
    test_bool_true_parsed();
    test_decoder_profile_selection_keeps_runtime_ceiling();
    test_dynamic_profile_rows_use_profile_metadata();
    test_exact_special_frame_overrides_native_unigram_fallback();
    test_exact_special_frame_respects_add_special_false();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All plugin helper tests passed.\n";
    return 0;
}
