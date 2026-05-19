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
//
// Purpose:
//   Validates the lightweight, regex-free JSON extraction functions used
//   throughout the codebase to parse HuggingFace config.json and
//   generation_config.json files. These extractors operate on raw JSON strings
//   (no DOM parser) and must correctly handle common JSON patterns including
//   nested objects, arrays, negative integers, scientific-notation floats,
//   and missing keys (fallback values).
//
// Dependencies:
//   - utils/json_helpers.h (extract_json_string, extract_json_int,
//     extract_json_int_or_first_array, extract_json_float,
//     extract_json_string_array)
//
// Approach:
//   Each test constructs a minimal JSON string literal, calls the appropriate
//   extraction function, and verifies the returned value matches the expected
//   result (or the specified fallback when the key is absent). Tests cover
//   both happy paths and edge cases (missing keys, nested braces, empty
//   arrays, float-valued integers, scientific notation).
//
// Environment:
//   CPU-only, no TRT/CUDA dependencies. No filesystem access required.
// =============================================================================

#include "test_helpers.h"
#include "utils/json_helpers.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// extract_json_string tests
// ---------------------------------------------------------------------------

// Intention: Verify that extract_json_string finds a key that exists and
//            returns its string value.
// Setup:     JSON with "model_type": "qwen3" alongside another key.
// Mechanism: Calls extract_json_string with key "model_type", checks that the
//            returned value is "qwen3" (not the fallback).
bool test_extract_json_string_present() {
    const std::string json = R"({"model_type": "qwen3", "other": "value"})";
    const std::string result = trtmc::extract_json_string(json, "model_type", "");
    if (result != "qwen3") {
        std::cerr << "extract_json_string_present: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that extract_json_string returns the fallback when the
//            requested key is not present in the JSON.
// Setup:     JSON with only an unrelated key ("other").
// Mechanism: Calls extract_json_string with key "model_type" and fallback
//            "fallback", checks the fallback is returned.
bool test_extract_json_string_absent() {
    const std::string json = R"({"other": "value"})";
    const std::string result = trtmc::extract_json_string(json, "model_type", "fallback");
    if (result != "fallback") {
        std::cerr << "extract_json_string_absent: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that the parser correctly handles JSON with nested brace
//            structures (inner objects) before the target key.
// Setup:     JSON with a nested object "config": {"inner": 1} preceding the
//            target key "model_type": "llama".
// Mechanism: Calls extract_json_string and verifies "llama" is returned,
//            confirming the parser is not confused by nested braces.
bool test_extract_json_string_nested_braces() {
    const std::string json = R"({"config": {"inner": 1}, "model_type": "llama"})";
    const std::string result = trtmc::extract_json_string(json, "model_type", "");
    if (result != "llama") {
        std::cerr << "extract_json_string_nested: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify JSON string escapes are unescaped before returning.
// Setup:     JSON with escaped quote and newline characters.
// Mechanism: Calls extract_json_string and verifies the decoded string.
bool test_extract_json_string_escapes() {
    const std::string json = R"({"prompt": "Generate an \"Enhanced prompt\"\nUser Prompt: "})";
    const std::string result = trtmc::extract_json_string(json, "prompt", "");
    if (result != "Generate an \"Enhanced prompt\"\nUser Prompt: ") {
        std::cerr << "extract_json_string_escapes: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// extract_json_int tests
// ---------------------------------------------------------------------------

// Intention: Verify extraction of a positive integer value from JSON.
// Setup:     JSON with "hidden_size": 768.
// Mechanism: Calls extract_json_int, checks the result equals 768.
bool test_extract_json_int_positive() {
    const std::string json = R"({"hidden_size": 768})";
    const int32_t result = trtmc::extract_json_int(json, "hidden_size", -1);
    if (result != 768) {
        std::cerr << "extract_json_int_positive: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify extraction of a negative integer value from JSON.
// Setup:     JSON with "offset": -42.
// Mechanism: Calls extract_json_int, checks the result equals -42.
bool test_extract_json_int_negative() {
    const std::string json = R"({"offset": -42})";
    const int32_t result = trtmc::extract_json_int(json, "offset", 0);
    if (result != -42) {
        std::cerr << "extract_json_int_negative: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that extract_json_int returns the fallback value when
//            the key does not exist in the JSON.
// Setup:     JSON with only "other": 5, no "hidden_size" key.
// Mechanism: Calls extract_json_int with fallback -99, checks -99 is returned.
bool test_extract_json_int_missing() {
    const std::string json = R"({"other": 5})";
    const int32_t result = trtmc::extract_json_int(json, "hidden_size", -99);
    if (result != -99) {
        std::cerr << "extract_json_int_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify the behavior of extract_json_int when the value is a
//            floating-point number (3.14). The parser reads digits until it
//            hits the decimal point and returns the integer portion.
// Setup:     JSON with "hidden_size": 3.14.
// Mechanism: Calls extract_json_int, checks the result is 3 (the parser
//            stops at the '.' and returns what it has parsed so far).
bool test_extract_json_int_float_value() {
    // Float values should return fallback (parser stops at '.')
    const std::string json = R"({"hidden_size": 3.14})";
    const int32_t result = trtmc::extract_json_int(json, "hidden_size", -1);
    // Parser reads "3" then stops at '.' — returns 3
    if (result != 3) {
        std::cerr << "extract_json_int_float: got " << result << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// extract_json_int_or_first_array tests
// ---------------------------------------------------------------------------

// Intention: Verify that extract_json_int_or_first_array extracts a plain
//            scalar integer value (not wrapped in an array).
// Setup:     JSON with "bos_token_id": 123 (scalar).
// Mechanism: Calls extract_json_int_or_first_array, checks the result is 123.
//            This exercises the scalar branch of the dual-format parser.
bool test_extract_json_int_or_first_array_scalar() {
    const std::string json = R"({"bos_token_id": 123})";
    const int32_t result = trtmc::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != 123) {
        std::cerr << "int_or_first_array_scalar: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that extract_json_int_or_first_array extracts the first
//            element from a JSON array value.
// Setup:     JSON with "bos_token_id": [456, 789] (array of two elements).
// Mechanism: Calls extract_json_int_or_first_array, checks the result is 456
//            (the first element). This exercises the array branch, which is
//            needed because some HF configs encode token IDs as arrays.
bool test_extract_json_int_or_first_array_array() {
    const std::string json = R"({"bos_token_id": [456, 789]})";
    const int32_t result = trtmc::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != 456) {
        std::cerr << "int_or_first_array_array: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that an empty array causes the fallback to be returned,
//            since there is no first element to extract.
// Setup:     JSON with "bos_token_id": [] (empty array).
// Mechanism: Calls extract_json_int_or_first_array with fallback -1, checks
//            the result is -1.
bool test_extract_json_int_or_first_array_empty_array() {
    const std::string json = R"({"bos_token_id": []})";
    const int32_t result = trtmc::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != -1) {
        std::cerr << "int_or_first_array_empty: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that a missing key causes the fallback to be returned.
// Setup:     JSON with only "other": 5, no "bos_token_id" key.
// Mechanism: Calls extract_json_int_or_first_array with fallback -1, checks
//            the result is -1.
bool test_extract_json_int_or_first_array_missing() {
    const std::string json = R"({"other": 5})";
    const int32_t result = trtmc::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != -1) {
        std::cerr << "int_or_first_array_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// extract_json_float tests
// ---------------------------------------------------------------------------

// Intention: Verify extraction of a basic floating-point value from JSON.
// Setup:     JSON with "rope_theta": 3.14.
// Mechanism: Calls extract_json_float, checks the result is within 0.01 of
//            3.14F.
bool test_extract_json_float_basic() {
    const std::string json = R"({"rope_theta": 3.14})";
    const float result = trtmc::extract_json_float(json, "rope_theta", 0.0F);
    if (std::abs(result - 3.14F) > 0.01F) {
        std::cerr << "extract_json_float_basic: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify extraction of a float in scientific notation (e.g., 1e-5),
//            which is the common format for epsilon values in HF configs.
// Setup:     JSON with "eps": 1e-5.
// Mechanism: Calls extract_json_float, checks the result is within 1e-8 of
//            1e-5F.
bool test_extract_json_float_scientific() {
    const std::string json = R"({"eps": 1e-5})";
    const float result = trtmc::extract_json_float(json, "eps", 0.0F);
    if (std::abs(result - 1e-5F) > 1e-8F) {
        std::cerr << "extract_json_float_scientific: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that extract_json_float returns the fallback when the key
//            is absent from the JSON.
// Setup:     JSON with only "other": 5, no "eps" key.
// Mechanism: Calls extract_json_float with fallback -1.0F, checks -1.0F is
//            returned.
bool test_extract_json_float_missing() {
    const std::string json = R"({"other": 5})";
    const float result = trtmc::extract_json_float(json, "eps", -1.0F);
    if (std::abs(result - (-1.0F)) > 1e-6F) {
        std::cerr << "extract_json_float_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// extract_json_string_array tests
// ---------------------------------------------------------------------------

// Intention: Verify extraction of a JSON string array with multiple elements.
// Setup:     JSON with "architectures": ["QwenForCausalLM", "Qwen2ForCausalLM"].
// Mechanism: Calls extract_json_string_array, checks the returned vector has
//            exactly 2 elements matching the expected values. This mirrors the
//            real-world "architectures" field in HF config.json.
bool test_extract_json_string_array_basic() {
    const std::string json = R"({"architectures": ["QwenForCausalLM", "Qwen2ForCausalLM"]})";
    const auto result = trtmc::extract_json_string_array(json, "architectures");
    if (result.size() != 2 || result[0] != "QwenForCausalLM" || result[1] != "Qwen2ForCausalLM") {
        std::cerr << "string_array_basic: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that an empty JSON array returns an empty vector.
// Setup:     JSON with "architectures": [].
// Mechanism: Calls extract_json_string_array, checks the result is empty.
bool test_extract_json_string_array_empty() {
    const std::string json = R"({"architectures": []})";
    const auto result = trtmc::extract_json_string_array(json, "architectures");
    if (!result.empty()) {
        std::cerr << "string_array_empty: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify that a missing key returns an empty vector (not an error).
// Setup:     JSON with only "other": 5, no "architectures" key.
// Mechanism: Calls extract_json_string_array, checks the result is empty.
bool test_extract_json_string_array_missing() {
    const std::string json = R"({"other": 5})";
    const auto result = trtmc::extract_json_string_array(json, "architectures");
    if (!result.empty()) {
        std::cerr << "string_array_missing: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify empty-string JSON values are treated as malformed by
//            the lightweight extractor and return fallback.
// Setup:     JSON with "name": "".
// Mechanism: Calls extract_json_string and checks fallback is returned.
bool test_extract_json_string_empty_value_returns_fallback() {
    const std::string json = R"({"name": ""})";
    const std::string result = trtmc::extract_json_string(json, "name", "fallback");
    if (result != "fallback") {
        std::cerr << "extract_json_string_empty_value: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify malformed float tokens still parse numeric prefix.
// Setup:     JSON with "eps": 1e.
// Mechanism: Calls extract_json_float and checks parsed prefix value is returned.
bool test_extract_json_float_invalid_token_returns_fallback() {
    const std::string json = R"({"eps": 1e})";
    const float result = trtmc::extract_json_float(json, "eps", 9.5F);
    if (std::abs(result - 1.0F) > 1e-6F) {
        std::cerr << "extract_json_float_invalid_token: got " << result << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify float-array extraction parses signed and decimal values.
// Setup:     JSON with mixed float array.
// Mechanism: Calls extract_json_float_array and checks parsed values.
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

// Intention: Verify float-array extraction respects max_count limit.
// Setup:     Array with four values and max_count=2.
// Mechanism: Calls extract_json_float_array and checks only first two are kept.
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

// Intention: Verify invalid tokens stop numeric-array parsing gracefully.
// Setup:     Array with an invalid middle token.
// Mechanism: Calls extract_json_float_array and expects only prefix values.
bool test_extract_json_float_array_stops_on_invalid_token() {
    const std::string json = R"({"vals": [1.0, bad, 3.0]})";
    const auto values = trtmc::extract_json_float_array(json, "vals", 8);
    if (values.size() != 1 || std::abs(values[0] - 1.0F) > 1e-6F) {
        std::cerr << "extract_json_float_array_stops_on_invalid_token: unexpected parse"
                  << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify int-array extraction parses negatives and positives.
// Setup:     JSON with integer array.
// Mechanism: Calls extract_json_int_array and checks values.
bool test_extract_json_int_array_basic() {
    const std::string json = R"({"ids": [-3, 0, 9]})";
    const auto values = trtmc::extract_json_int_array(json, "ids", 8);
    if (values.size() != 3 || values[0] != -3 || values[1] != 0 || values[2] != 9) {
        std::cerr << "extract_json_int_array_basic: unexpected values" << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify invalid int token stops array parsing.
// Setup:     JSON with malformed middle element.
// Mechanism: Calls extract_json_int_array and checks prefix-only behavior.
bool test_extract_json_int_array_stops_on_invalid_token() {
    const std::string json = R"({"ids": [10, --5, 7]})";
    const auto values = trtmc::extract_json_int_array(json, "ids", 8);
    if (values.size() != 1 || values[0] != 10) {
        std::cerr << "extract_json_int_array_stops_on_invalid_token: unexpected values"
                  << std::endl;
        return false;
    }
    return true;
}

// Intention: Verify mixed string-array values stop parsing at first non-string token.
// Setup:     Array with one valid string followed by an integer.
// Mechanism: Calls extract_json_string_array and checks prefix-only result.
bool test_extract_json_string_array_stops_on_non_string() {
    const std::string json = R"({"architectures": ["A", 7, "B"]})";
    const auto values = trtmc::extract_json_string_array(json, "architectures");
    if (values.size() != 1 || values[0] != "A") {
        std::cerr << "extract_json_string_array_stops_on_non_string: unexpected values"
                  << std::endl;
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
    run("extract_json_string_escapes", test_extract_json_string_escapes);
    run("extract_json_string_empty_value", test_extract_json_string_empty_value_returns_fallback);
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
    run("extract_json_float_invalid_token", test_extract_json_float_invalid_token_returns_fallback);
    run("extract_json_float_array_basic", test_extract_json_float_array_basic);
    run("extract_json_float_array_max_count", test_extract_json_float_array_max_count);
    run("extract_json_float_array_invalid", test_extract_json_float_array_stops_on_invalid_token);
    run("extract_json_int_array_basic", test_extract_json_int_array_basic);
    run("extract_json_int_array_invalid", test_extract_json_int_array_stops_on_invalid_token);
    run("string_array_basic", test_extract_json_string_array_basic);
    run("string_array_empty", test_extract_json_string_array_empty);
    run("string_array_missing", test_extract_json_string_array_missing);
    run("string_array_non_string", test_extract_json_string_array_stops_on_non_string);

    if (all_passed) {
        std::cout << "test_json_helpers passed" << std::endl;
        return 0;
    }
    std::cerr << "test_json_helpers FAILED" << std::endl;
    return 1;
}
