/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-UTIL-CPP-01
// Architecture:   ARCH-BDL-001
// Unit Design:    UD-UTIL-01
// Intent:         JSON extraction helpers (string, int, float, array)
// Preconditions:  Valid and invalid JSON strings
// Postconditions: Correct values extracted, fallbacks used for missing keys
// =============================================================================

// =============================================================================
// test_json_helpers.cpp — Unit tests for src/utils/json_helpers.cpp
// =============================================================================

#include "test_helpers.h"
#include "utils/json_helpers.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool test_extract_json_string_present() {
    const std::string json = R"({"model_type": "qwen3", "other": "value"})";
    const std::string result = trtmc::extract_json_string(json, "model_type", "");
    if (result != "qwen3") {
        std::cerr << "extract_json_string_present: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_absent() {
    const std::string json = R"({"other": "value"})";
    const std::string result = trtmc::extract_json_string(json, "model_type", "fallback");
    if (result != "fallback") {
        std::cerr << "extract_json_string_absent: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_nested_braces() {
    const std::string json = R"({"config": {"inner": 1}, "model_type": "decoder"})";
    const std::string result = trtmc::extract_json_string(json, "model_type", "");
    if (result != "decoder") {
        std::cerr << "extract_json_string_nested: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_escapes_and_unicode() {
    const std::string json = R"({"prompt":"schema: {\"name\": \"caf\u00e9\"}\nnext \ud83d\ude80"})";
    const std::string expected = "schema: {\"name\": \"caf\xC3\xA9\"}\nnext \xF0\x9F\x9A\x80";
    const std::string result = trtmc::extract_json_string(json, "prompt", "fallback");
    if (result != expected) {
        std::cerr << "extract_json_string_escapes: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_positive() {
    const std::string json = R"({"hidden_size": 768})";
    const int32_t result = trtmc::extract_json_int(json, "hidden_size", -1);
    if (result != 768) {
        std::cerr << "extract_json_int_positive: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_negative() {
    const std::string json = R"({"offset": -42})";
    const int32_t result = trtmc::extract_json_int(json, "offset", 0);
    if (result != -42) {
        std::cerr << "extract_json_int_negative: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_missing() {
    const std::string json = R"({"other": 5})";
    const int32_t result = trtmc::extract_json_int(json, "hidden_size", -99);
    if (result != -99) {
        std::cerr << "extract_json_int_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_float_value() {
    const std::string json = R"({"hidden_size": 3.14})";
    const int32_t result = trtmc::extract_json_int(json, "hidden_size", -1);
    if (result != 3) {
        std::cerr << "extract_json_int_float: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_or_first_array_scalar() {
    const std::string json = R"({"bos_token_id": 123})";
    const int32_t result = trtmc::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != 123) {
        std::cerr << "int_or_first_array_scalar: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_or_first_array_array() {
    const std::string json = R"({"bos_token_id": [456, 789]})";
    const int32_t result = trtmc::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != 456) {
        std::cerr << "int_or_first_array_array: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_or_first_array_empty_array() {
    const std::string json = R"({"bos_token_id": []})";
    const int32_t result = trtmc::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != -1) {
        std::cerr << "int_or_first_array_empty: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_or_first_array_missing() {
    const std::string json = R"({"other": 5})";
    const int32_t result = trtmc::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != -1) {
        std::cerr << "int_or_first_array_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_float_basic() {
    const std::string json = R"({"rope_theta": 3.14})";
    const float result = trtmc::extract_json_float(json, "rope_theta", 0.0F);
    if (std::abs(result - 3.14F) > 0.01F) {
        std::cerr << "extract_json_float_basic: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_float_scientific() {
    const std::string json = R"({"eps": 1e-5})";
    const float result = trtmc::extract_json_float(json, "eps", 0.0F);
    if (std::abs(result - 1e-5F) > 1e-8F) {
        std::cerr << "extract_json_float_scientific: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_float_missing() {
    const std::string json = R"({"other": 5})";
    const float result = trtmc::extract_json_float(json, "eps", -1.0F);
    if (std::abs(result - (-1.0F)) > 1e-6F) {
        std::cerr << "extract_json_float_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_array_basic() {
    const std::string json = R"({"architectures": ["QwenForCausalLM", "Qwen2ForCausalLM"]})";
    const auto result = trtmc::extract_json_string_array(json, "architectures");
    if (result.size() != 2 || result[0] != "QwenForCausalLM" || result[1] != "Qwen2ForCausalLM") {
        std::cerr << "string_array_basic: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_array_empty() {
    const std::string json = R"({"architectures": []})";
    const auto result = trtmc::extract_json_string_array(json, "architectures");
    if (!result.empty()) {
        std::cerr << "string_array_empty: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_array_missing() {
    const std::string json = R"({"other": 5})";
    const auto result = trtmc::extract_json_string_array(json, "architectures");
    if (!result.empty()) {
        std::cerr << "string_array_missing: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Updated Contract Tests (Strict Standard JSON Compliance)
// ---------------------------------------------------------------------------

// Empty strings are now valid JSON strings, so they should be extracted successfully.
bool test_extract_json_string_empty_value_returns_empty() {
    const std::string json = R"({"name": ""})";
    const std::string result = trtmc::extract_json_string(json, "name", "fallback");
    if (result != "") {
        std::cerr << "extract_json_string_empty_value: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

// Malformed tokens should fail the entire document parsing and return fallback.
bool test_extract_json_float_invalid_token_returns_fallback() {
    const std::string json = R"({"eps": 1e})";
    const float result = trtmc::extract_json_float(json, "eps", 9.5F);
    if (std::abs(result - 9.5F) > 1e-6F) {
        std::cerr << "extract_json_float_invalid_token: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_float_array_basic() {
    const std::string json = R"({"image_mean": [0.5, -1.25, 2.0]})";
    const auto values = trtmc::extract_json_float_array(json, "image_mean", 8);
    if (values.size() != 3) {
        std::cerr << "extract_json_float_array_basic: size=" << values.size() << std::endl;
        return false;
    }
    if (std::abs(values[0] - 0.5F) > 1e-6F || std::abs(values[1] + 1.25F) > 1e-6F ||
        std::abs(values[2] - 2.0F) > 1e-6F) {
        std::cerr << "extract_json_float_array_basic: value mismatch" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_float_array_max_count() {
    const std::string json = R"({"vals": [1.0, 2.0, 3.0, 4.0]})";
    const auto values = trtmc::extract_json_float_array(json, "vals", 2);
    if (values.size() != 2 || std::abs(values[0] - 1.0F) > 1e-6F ||
        std::abs(values[1] - 2.0F) > 1e-6F) {
        std::cerr << "extract_json_float_array_max_count: unexpected values" << std::endl;
        return false;
    }
    return true;
}

// Invalid token in an array malforms the JSON. Fallback is empty array.
bool test_extract_json_float_array_stops_on_invalid_token() {
    const std::string json = R"({"vals": [1.0, bad, 3.0]})";
    const auto values = trtmc::extract_json_float_array(json, "vals", 8);
    if (!values.empty()) {
        std::cerr << "extract_json_float_array_stops_on_invalid_token: unexpected parse"
                  << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_array_basic() {
    const std::string json = R"({"ids": [-3, 0, 9]})";
    const auto values = trtmc::extract_json_int_array(json, "ids", 8);
    if (values.size() != 3 || values[0] != -3 || values[1] != 0 || values[2] != 9) {
        std::cerr << "extract_json_int_array_basic: unexpected values" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_array_rejects_scalar() {
    const std::string json = R"({"eos_token_id": 2, "other_ids": [5, 7]})";
    const auto values = trtmc::extract_json_int_array(json, "eos_token_id", 8);
    if (!values.empty()) {
        std::cerr << "extract_json_int_array_rejects_scalar: unexpected values" << std::endl;
        return false;
    }
    return true;
}

// Invalid token in an array malforms the JSON. Fallback is empty array.
bool test_extract_json_int_array_stops_on_invalid_token() {
    const std::string json = R"({"ids": [10, --5, 7]})";
    const auto values = trtmc::extract_json_int_array(json, "ids", 8);
    if (!values.empty()) {
        std::cerr << "extract_json_int_array_stops_on_invalid_token: unexpected values"
                  << std::endl;
        return false;
    }
    return true;
}

// Array containing non-strings just skips them rather than stopping completely.
bool test_extract_json_string_array_skips_non_string() {
    const std::string json = R"({"architectures": ["A", 7, "B"]})";
    const auto values = trtmc::extract_json_string_array(json, "architectures");
    if (values.size() != 2 || values[0] != "A" || values[1] != "B") {
        std::cerr << "extract_json_string_array_skips_non_string: unexpected values"
                  << std::endl;
        return false;
    }
    return true;
}

// Duplicate key test. nlohmann/json standard behavior is to retain the last value.
bool test_duplicate_key_returns_last() {
    const std::string json = R"({"key": 1, "key": 2})";
    const int32_t result = trtmc::extract_json_int(json, "key", 0);
    if (result != 2) {
        std::cerr << "test_duplicate_key_returns_last: got " << result << std::endl;
        return false;
    }
    return true;
}

// Comments are ignored.
bool test_json_ignores_comments() {
    const std::string json = R"({"key": /* comment */ 42})";
    const int32_t result = trtmc::extract_json_int(json, "key", 0);
    if (result != 42) {
        std::cerr << "test_json_ignores_comments: got " << result << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main() {
    bool all_passed = true;
    std::cout << "test_json_helpers:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("extract_json_string_present", test_extract_json_string_present);
    run("extract_json_string_absent", test_extract_json_string_absent);
    run("extract_json_string_nested", test_extract_json_string_nested_braces);
    run("extract_json_string_escapes", test_extract_json_string_escapes_and_unicode);
    run("extract_json_int_positive", test_extract_json_int_positive);
    run("extract_json_int_negative", test_extract_json_int_negative);
    run("extract_json_int_missing", test_extract_json_int_missing);
    run("extract_json_int_float_value", test_extract_json_int_float_value);
    run("int_or_first_array_scalar", test_extract_json_int_or_first_array_scalar);
    run("int_or_first_array_array", test_extract_json_int_or_first_array_array);
    run("int_or_first_array_empty", test_extract_json_int_or_first_array_empty_array);
    run("int_or_first_array_missing", test_extract_json_int_or_first_array_missing);
    run("extract_json_float_basic", test_extract_json_float_basic);
    run("extract_json_float_scientific", test_extract_json_float_scientific);
    run("extract_json_float_missing", test_extract_json_float_missing);
    run("extract_json_string_array_basic", test_extract_json_string_array_basic);
    run("extract_json_string_array_empty", test_extract_json_string_array_empty);
    run("extract_json_string_array_missing", test_extract_json_string_array_missing);
    run("extract_json_string_empty_value", test_extract_json_string_empty_value_returns_empty);
    run("extract_json_float_invalid_token", test_extract_json_float_invalid_token_returns_fallback);
    run("extract_json_float_array_basic", test_extract_json_float_array_basic);
    run("extract_json_float_array_max_count", test_extract_json_float_array_max_count);
    run("extract_json_float_array_invalid", test_extract_json_float_array_stops_on_invalid_token);
    run("extract_json_int_array_basic", test_extract_json_int_array_basic);
    run("extract_json_int_array_rejects_scalar", test_extract_json_int_array_rejects_scalar);
    run("extract_json_int_array_invalid", test_extract_json_int_array_stops_on_invalid_token);
    run("string_array_skips_non_string", test_extract_json_string_array_skips_non_string);
    run("duplicate_key_returns_last", test_duplicate_key_returns_last);
    run("json_ignores_comments", test_json_ignores_comments);

    if (all_passed) {
        std::cout << "test_json_helpers passed" << std::endl;
        return 0;
    }
    std::cerr << "test_json_helpers FAILED" << std::endl;
    return 1;
}
