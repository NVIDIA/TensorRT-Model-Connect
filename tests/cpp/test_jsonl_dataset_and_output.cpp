/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CLI-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CABI-01
// Intent:         JSONL dataset parsing and output record construction
// Preconditions:  None
// Postconditions: parse_dataset_line correctly decodes valid records and rejects
//                  malformed input; build_*_record functions produce valid JSON
//                  with the correct field names, types, and values.
// =============================================================================

// =============================================================================
// test_jsonl_dataset_and_output.cpp — Unit tests for src/cli/jsonl_io.{h,cpp}
// =============================================================================
//
// Purpose:
//   Validates the JSONL dataset line parser and JSON record builders used by
//   the trtmc CLI and the trtmc_dataset_benchmark tool. Tests cover:
//   - Quotes, backslashes, control characters, and Unicode in string fields
//   - Missing required fields (sample_id, answer, prompt)
//   - Wrong field types (e.g. integer sample_id, string seed_index)
//   - Optional seed_index int32 boundaries and overflow rejection
//   - Malformed JSONL records (truncated, unbalanced, trailing commas)
//   - Optional seed_index presence and absence
//   - Output record round-trip (dump -> parse -> field name/value/type match)
//   - Non-finite float detection in tensor records
//
// Environment:
//   CPU-only, no TRT/CUDA dependencies. No filesystem access required.
// =============================================================================

#include "cli/jsonl_io.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    } else {
        std::cout << "  " << test_name << ": PASS\n";
    }
}

template <typename Function>
static void check_throws(const char* test_name, Function&& fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, test_name);
}

// ---------------------------------------------------------------------------
// parse_dataset_line tests
// ---------------------------------------------------------------------------

namespace {

bool test_parse_basic() {
    const std::string line = R"({"sample_id":"q1","answer":"42","prompt":"What is 6*7?"})";
    auto s = trtmc::cli::parse_dataset_line(line, 1);
    return s.sample_id == "q1" && s.answer == "42" && s.prompt == "What is 6*7?" &&
           !s.seed_index.has_value();
}

bool test_parse_with_seed_index() {
    const std::string line = R"({"sample_id":"q2","answer":"7","prompt":"3+4","seed_index":5})";
    auto s = trtmc::cli::parse_dataset_line(line, 1);
    return s.seed_index.has_value() && s.seed_index.value() == 5;
}

bool test_parse_seed_index_int32_boundaries() {
    const auto minimum = trtmc::cli::parse_dataset_line(
        R"({"sample_id":"min","answer":"ok","prompt":"test","seed_index":-2147483648})", 1);
    const auto maximum = trtmc::cli::parse_dataset_line(
        R"({"sample_id":"max","answer":"ok","prompt":"test","seed_index":2147483647})", 1);
    return minimum.seed_index == std::numeric_limits<int32_t>::min() &&
           maximum.seed_index == std::numeric_limits<int32_t>::max();
}

bool test_parse_seed_index_overflow() {
    const std::vector<std::string> lines = {
        R"({"sample_id":"high","answer":"ok","prompt":"test","seed_index":2147483648})",
        R"({"sample_id":"low","answer":"ok","prompt":"test","seed_index":-2147483649})",
        R"({"sample_id":"huge","answer":"ok","prompt":"test","seed_index":18446744073709551615})",
    };
    for (const auto& line : lines) {
        try {
            trtmc::cli::parse_dataset_line(line, 23);
            return false;
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            if (message.find("seed_index") == std::string::npos ||
                message.find("int32 range") == std::string::npos ||
                message.find("23") == std::string::npos) {
                return false;
            }
        }
    }
    return true;
}

bool test_parse_seed_index_float_is_not_an_integer() {
    try {
        trtmc::cli::parse_dataset_line(
            R"({"sample_id":"float","answer":"ok","prompt":"test","seed_index":1.5})", 9);
        return false;
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        return message.find("seed_index") != std::string::npos &&
               message.find("integer") != std::string::npos &&
               message.find("9") != std::string::npos;
    }
}

bool test_parse_with_quotes_and_backslashes() {
    const std::string line =
        R"({"sample_id":"q3","answer":"yes","prompt":"She said \"hello\\world\""})";
    auto s = trtmc::cli::parse_dataset_line(line, 1);
    return s.prompt == R"(She said "hello\world")";
}

bool test_parse_with_control_characters() {
    const std::string line =
        R"({"sample_id":"q4","answer":"ok","prompt":"line1\nline2\ttab\rreturn"})";
    auto s = trtmc::cli::parse_dataset_line(line, 1);
    return s.prompt == "line1\nline2\ttab\rreturn";
}

bool test_parse_with_unicode() {
    // \u00e9 = é, \ud83d\ude80 = 🚀
    const std::string line =
        R"({"sample_id":"q5","answer":"ok","prompt":"caf\u00e9 \ud83d\ude80"})";
    auto s = trtmc::cli::parse_dataset_line(line, 1);
    return s.prompt == "caf\xC3\xA9 \xF0\x9F\x9A\x80";
}

bool test_parse_missing_sample_id() {
    const std::string line = R"({"answer":"42","prompt":"test"})";
    try {
        trtmc::cli::parse_dataset_line(line, 7);
        return false;
    } catch (const std::runtime_error& e) {
        // Must mention missing field and line number
        std::string msg = e.what();
        return msg.find("sample_id") != std::string::npos && msg.find("7") != std::string::npos;
    }
}

bool test_parse_missing_answer() {
    const std::string line = R"({"sample_id":"q1","prompt":"test"})";
    try {
        trtmc::cli::parse_dataset_line(line, 3);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("answer") != std::string::npos;
    }
}

bool test_parse_missing_prompt() {
    const std::string line = R"({"sample_id":"q1","answer":"42"})";
    try {
        trtmc::cli::parse_dataset_line(line, 4);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("prompt") != std::string::npos;
    }
}

bool test_parse_wrong_type_sample_id() {
    // sample_id is an integer instead of a string
    const std::string line = R"({"sample_id":123,"answer":"42","prompt":"test"})";
    try {
        trtmc::cli::parse_dataset_line(line, 1);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("sample_id") != std::string::npos &&
               msg.find("string") != std::string::npos;
    }
}

bool test_parse_wrong_type_seed_index() {
    // seed_index is a string instead of integer — should throw
    const std::string line =
        R"({"sample_id":"q1","answer":"42","prompt":"test","seed_index":"not_int"})";
    try {
        trtmc::cli::parse_dataset_line(line, 1);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("seed_index") != std::string::npos &&
               msg.find("integer") != std::string::npos;
    }
}

bool test_parse_malformed_json() {
    const std::string line = R"({"sample_id":"q1","answer":"42","prompt":)";
    try {
        trtmc::cli::parse_dataset_line(line, 42);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("42") != std::string::npos &&
               msg.find("Malformed JSON") != std::string::npos;
    }
}

bool test_parse_not_an_object() {
    const std::string line = R"([1, 2, 3])";
    try {
        trtmc::cli::parse_dataset_line(line, 1);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        return msg.find("object") != std::string::npos;
    }
}

bool test_multiline_dataset_processing() {
    const std::vector<std::string> lines = {
        R"({"sample_id":"s1","answer":"ans1","prompt":"p1"})",
        "", // empty line to skip
        R"({"sample_id":"s2","answer":"ans2","prompt":"p2","seed_index":"bad_type"})"};

    std::size_t line_no = 0;
    std::vector<trtmc::cli::DatasetSample> parsed_samples;
    bool caught_expected_error = false;

    for (const auto& l : lines) {
        ++line_no;
        if (l.empty())
            continue;
        try {
            parsed_samples.push_back(trtmc::cli::parse_dataset_line(l, line_no));
        } catch (const std::runtime_error& e) {
            std::string msg = e.what();
            if (line_no == 3 && msg.find("line 3") != std::string::npos &&
                msg.find("seed_index") != std::string::npos) {
                caught_expected_error = true;
            }
            break;
        }
    }

    return parsed_samples.size() == 1 && parsed_samples[0].sample_id == "s1" &&
           caught_expected_error;
}

// ---------------------------------------------------------------------------
// build_text_sample_record tests
// ---------------------------------------------------------------------------

bool test_build_text_sample_record_roundtrip() {
    trtmc::TextResult result;
    result.text = "Paris";
    result.token_ids = {101, 202, 303};

    auto record = trtmc::cli::build_text_sample_record(42, "What is the capital?", result);
    auto parsed = nlohmann::json::parse(record.dump());

    return parsed["id"].get<int32_t>() == 42 &&
           parsed["prompt"].get<std::string>() == "What is the capital?" &&
           parsed["generated"].get<std::string>() == "Paris" && parsed["token_ids"].size() == 3 &&
           parsed["token_ids"][0].get<int32_t>() == 101 &&
           parsed["token_ids"][1].get<int32_t>() == 202 &&
           parsed["token_ids"][2].get<int32_t>() == 303;
}

bool test_build_text_sample_record_special_chars() {
    trtmc::TextResult result;
    result.text = "line1\nline2\t\"quoted\\back\"";
    result.token_ids = {1};

    auto record = trtmc::cli::build_text_sample_record(0, "prompt with\nnewline", result);
    auto parsed = nlohmann::json::parse(record.dump());

    return parsed["prompt"].get<std::string>() == "prompt with\nnewline" &&
           parsed["generated"].get<std::string>() == "line1\nline2\t\"quoted\\back\"";
}

bool test_build_text_sample_record_empty_tokens() {
    trtmc::TextResult result;
    result.text = "";
    result.token_ids = {};

    auto record = trtmc::cli::build_text_sample_record(0, "", result);
    auto parsed = nlohmann::json::parse(record.dump());

    return parsed["token_ids"].is_array() && parsed["token_ids"].empty();
}

// ---------------------------------------------------------------------------
// build_classify_record tests
// ---------------------------------------------------------------------------

bool test_build_classify_record_roundtrip() {
    trtmc::ClassificationResult result;
    result.top_class = 5;
    result.top_score = 0.95F;
    result.logits.resize(10, 0.0F);

    auto record = trtmc::cli::build_classify_record(result);
    auto parsed = nlohmann::json::parse(record.dump());

    return parsed["top_class"].get<int32_t>() == 5 &&
           std::abs(parsed["top_score"].get<float>() - 0.95F) < 1e-6F &&
           parsed["num_classes"].get<std::size_t>() == 10;
}

// ---------------------------------------------------------------------------
// build_tensor_record tests
// ---------------------------------------------------------------------------

bool test_build_tensor_record_roundtrip() {
    std::vector<int64_t> shape = {2, 3};
    std::vector<float> data = {1.0F, 2.5F, 0.0F, -1.0F, 3.14F, 100.0F};

    auto record = trtmc::cli::build_tensor_record(shape, data);
    auto parsed = nlohmann::json::parse(record.dump());

    if (parsed["shape"].size() != 2 || parsed["shape"][0].get<int64_t>() != 2 ||
        parsed["shape"][1].get<int64_t>() != 3)
        return false;
    if (parsed["data"].size() != 6)
        return false;
    // Check numeric values round-trip correctly
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (std::abs(parsed["data"][i].get<float>() - data[i]) > 1e-6F)
            return false;
    }
    return true;
}

bool test_build_tensor_record_non_finite() {
    std::vector<int64_t> shape = {1};
    std::vector<float> data_inf = {std::numeric_limits<float>::infinity()};
    std::vector<float> data_nan = {std::numeric_limits<float>::quiet_NaN()};

    bool caught_inf = false;
    try {
        trtmc::cli::build_tensor_record(shape, data_inf);
    } catch (const std::runtime_error& e) {
        caught_inf = std::string(e.what()).find("non-finite") != std::string::npos;
    }

    bool caught_nan = false;
    try {
        trtmc::cli::build_tensor_record(shape, data_nan);
    } catch (const std::runtime_error& e) {
        caught_nan = std::string(e.what()).find("non-finite") != std::string::npos;
    }

    return caught_inf && caught_nan;
}

bool test_build_tensor_record_empty() {
    std::vector<int64_t> shape = {0};
    std::vector<float> data = {};

    auto record = trtmc::cli::build_tensor_record(shape, data);
    auto parsed = nlohmann::json::parse(record.dump());

    return parsed["data"].is_array() && parsed["data"].empty();
}

// ---------------------------------------------------------------------------
// build_image_features_record tests
// ---------------------------------------------------------------------------

bool test_build_image_features_record_roundtrip() {
    trtmc::ImageFeaturesResult result;
    result.last_hidden_state = {1.0F, 2.0F};
    result.last_hidden_state_shape = {1, 2};
    result.pooler_output = {3.0F};
    result.pooler_output_shape = {1};

    auto record = trtmc::cli::build_image_features_record(result);
    auto parsed = nlohmann::json::parse(record.dump());

    return parsed.contains("last_hidden_state") && parsed.contains("pooler_output") &&
           parsed["last_hidden_state"]["shape"][0].get<int64_t>() == 1 &&
           parsed["last_hidden_state"]["data"].size() == 2 &&
           parsed["pooler_output"]["data"].size() == 1;
}

} // namespace

int main() {
    std::cout << "test_jsonl_dataset_and_output:" << std::endl;

    // parse_dataset_line
    check(test_parse_basic(), "parse_basic");
    check(test_parse_with_seed_index(), "parse_with_seed_index");
    check(test_parse_seed_index_int32_boundaries(), "parse_seed_index_int32_boundaries");
    check(test_parse_seed_index_overflow(), "parse_seed_index_overflow");
    check(test_parse_seed_index_float_is_not_an_integer(), "parse_seed_index_float");
    check(test_parse_with_quotes_and_backslashes(), "parse_quotes_backslashes");
    check(test_parse_with_control_characters(), "parse_control_characters");
    check(test_parse_with_unicode(), "parse_unicode");
    check(test_parse_missing_sample_id(), "parse_missing_sample_id");
    check(test_parse_missing_answer(), "parse_missing_answer");
    check(test_parse_missing_prompt(), "parse_missing_prompt");
    check(test_parse_wrong_type_sample_id(), "parse_wrong_type_sample_id");
    check(test_parse_wrong_type_seed_index(), "parse_wrong_type_seed_index");
    check(test_parse_malformed_json(), "parse_malformed_json");
    check(test_parse_not_an_object(), "parse_not_an_object");
    check(test_multiline_dataset_processing(), "multiline_dataset_processing");

    // build_text_sample_record
    check(test_build_text_sample_record_roundtrip(), "build_text_sample_roundtrip");
    check(test_build_text_sample_record_special_chars(), "build_text_sample_special_chars");
    check(test_build_text_sample_record_empty_tokens(), "build_text_sample_empty_tokens");

    // build_classify_record
    check(test_build_classify_record_roundtrip(), "build_classify_roundtrip");

    // build_tensor_record
    check(test_build_tensor_record_roundtrip(), "build_tensor_roundtrip");
    check(test_build_tensor_record_non_finite(), "build_tensor_non_finite");
    check(test_build_tensor_record_empty(), "build_tensor_empty");

    // build_image_features_record
    check(test_build_image_features_record_roundtrip(), "build_image_features_roundtrip");

    if (failures == 0) {
        std::cout << "test_jsonl_dataset_and_output passed" << std::endl;
        return EXIT_SUCCESS;
    }
    std::cerr << "test_jsonl_dataset_and_output FAILED (" << failures << " failures)" << std::endl;
    return EXIT_FAILURE;
}
