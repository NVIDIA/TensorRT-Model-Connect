/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-ABI-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CABI-01
// Intent:         C ABI runtime regression: invalid engine plans report errors without crashing
// Preconditions:  Syntactically valid .bundle with invalid engine_plan payload
// Postconditions: trtmc_last_error() returns descriptive message, no crash on repeated calls
// =============================================================================

// =============================================================================
// C ABI runtime regression tests for bundle -> TRT runtime -> deserialize path.
//
// These tests build a syntactically valid .bundle file with an invalid
// engine_plan payload, then call trtmc_create_pipeline repeatedly. This catches
// crashes in runtime/logger lifetime and validates that failures are reported
// through trtmc_last_error().
// =============================================================================

#include "test_helpers.h"
#include "trtmc/pipeline.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void write_u64_le(std::ofstream& out, uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFFU);
    }
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

struct BundleSectionSpec {
    std::string name;
    std::string data;
};

std::string build_bundle_header_json(const std::vector<BundleSectionSpec>& sections,
                                     const std::string& model_id) {
    std::string sections_json;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < sections.size(); ++i) {
        const auto& section = sections[i];
        if (i != 0) {
            sections_json += ",\n";
        }
        sections_json += "    \"" + section.name + "\": {\"offset\": " + std::to_string(offset) +
                         ", \"size\": " + std::to_string(section.data.size()) + "}";
        offset += section.data.size();
    }

    return std::string(R"({
  "model_id": ")") +
           model_id + R"(",
  "model_type": "unit-test",
  "family": "unit",
  "hidden_size": 64,
  "num_layers": 1,
  "num_attention_heads": 1,
  "num_key_value_heads": 1,
  "max_cache_length": 32,
  "sections": {
)" + sections_json +
           R"(
  }
})";
}

void write_bundle_with_sections(const std::filesystem::path& path,
                                const std::vector<BundleSectionSpec>& sections,
                                const std::string& model_id = "runtime-regression-test") {
    // Internal .bundle magic: "BUNDLE\x01\x00"
    static constexpr unsigned char kBundleMagic[8] = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', '\0'};

    const std::string header = build_bundle_header_json(sections, model_id);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(kBundleMagic), sizeof(kBundleMagic));
    write_u64_le(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& section : sections) {
        out.write(section.data.data(), static_cast<std::streamsize>(section.data.size()));
    }
}

void write_invalid_engine_bundle(const std::filesystem::path& path) {
    // Intentionally invalid TensorRT plan payload.
    static constexpr char kInvalidPlan[16] = {'N', 'O', 'T', '_', 'A', '_', 'P', 'L',
                                              'A', 'N', '_', 'B', 'L', 'O', 'B', '!'};
    const std::string config = R"({
  "runtime_strategy": "qwen_decoder_kv_cache",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(
        path,
        {BundleSectionSpec{"config.json", config},
         BundleSectionSpec{"engine_plan", std::string(kInvalidPlan, sizeof(kInvalidPlan))}},
        "qwen");
}

bool message_contains_any(const std::string& msg, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (msg.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool message_contains_any_expected_failure(const std::string& msg) {
    return message_contains_any(
        msg,
        {"deserialize engine", "deserialize failed", "execution context", "Failed to load bundle",
         "New runtime build failed", "Failed to deserialize engine", "Bundle missing engine plan",
         "Bundle missing", "No plugin registered", "Backend \"trt\" not available",
         "Could not load libtrtmc_backend_trt.so", "No compatible backend DSO available"});
}

void expect_invalid_bundle_creation_fails(const std::string& bundle_path, const char* test_name) {
    auto* pipeline = trtmc_create_pipeline(bundle_path.c_str(), 0);
    check(pipeline == nullptr, test_name);

    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "trtmc_last_error set for invalid plan bundle");
    if (err != nullptr) {
        check(message_contains_any_expected_failure(err),
              "error message indicates TRT runtime failure");
    }
}

void test_invalid_plan_bundle_reports_error() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path bundle_path =
        std::filesystem::path(dir.path()) / "invalid_engine_plan.bundle";
    write_invalid_engine_bundle(bundle_path);
    expect_invalid_bundle_creation_fails(bundle_path.string(),
                                         "invalid plan bundle returns nullptr");
}

void test_missing_engine_plan_bundle_reports_error() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path bundle_path =
        std::filesystem::path(dir.path()) / "missing_engine_plan.bundle";

    // Preconditions: valid bundle + model-owned config.json section, but no
    // engine_plan section.
    const std::string config = R"({
  "runtime_strategy": "qwen_decoder_kv_cache",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(bundle_path, {BundleSectionSpec{"config.json", config}}, "qwen");

    auto* pipeline = trtmc_create_pipeline(bundle_path.string().c_str(), 0);
    check(pipeline == nullptr, "bundle missing engine_plan returns nullptr");

    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "trtmc_last_error set for missing engine_plan");
    if (err != nullptr) {
        check(message_contains_any(err, {"New runtime build failed", "engine_plan",
                                         "Bundle missing engine plan", "Bundle missing",
                                         "Backend \"trt\" not available",
                                         "Could not load libtrtmc_backend_trt.so",
                                         "No compatible backend DSO available"}),
              "migrated strategy defaults to new runtime and reports engine_plan guard");
    }
}

void test_unknown_strategy_reports_new_runtime_unsupported_strategy_error() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path bundle_path =
        std::filesystem::path(dir.path()) / "unknown_strategy_missing_engine_plan.bundle";

    const std::string config = R"({
  "runtime_strategy": "future_unknown_strategy",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(bundle_path, {BundleSectionSpec{"config.json", config}});

    auto* pipeline = trtmc_create_pipeline(bundle_path.string().c_str(), 0);
    check(pipeline == nullptr, "unknown strategy without engine_plan returns nullptr");

    const char* err = trtmc_last_error();
    check(err != nullptr && std::strlen(err) > 0, "trtmc_last_error set for unknown strategy");
    if (err != nullptr) {
        check(message_contains_any(err,
                                   {"Unsupported runtime_strategy for new runtime path",
                                    "Unsupported audio strategy", "Unsupported text strategy",
                                    "Unsupported vision strategy", "Unsupported encoder strategy",
                                    "Unsupported diffusion strategy", "Unknown runtime_strategy",
                                    "No plugin registered"}),
              "unknown strategy fails through the new runtime unsupported-strategy guard");
    }
}

void test_invalid_plan_bundle_repeatable() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path bundle_path =
        std::filesystem::path(dir.path()) / "invalid_engine_plan_loop.bundle";
    write_invalid_engine_bundle(bundle_path);

    for (int i = 0; i < 25; ++i) {
        expect_invalid_bundle_creation_fails(bundle_path.string(),
                                             "invalid plan bundle repeated returns nullptr");
    }
}

} // namespace

int main() {
    test_invalid_plan_bundle_reports_error();
    test_missing_engine_plan_bundle_reports_error();
    test_unknown_strategy_reports_new_runtime_unsupported_strategy_error();
    test_invalid_plan_bundle_repeatable();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI runtime regression tests passed.\n";
    return 0;
}
