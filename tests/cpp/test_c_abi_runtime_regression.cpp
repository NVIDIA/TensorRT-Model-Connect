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
// Preconditions:  Syntactically valid .trtfb with invalid engine_plan payload
// Postconditions: trtmc_last_error() returns descriptive message, no crash on repeated calls
// =============================================================================

// =============================================================================
// C ABI runtime regression tests for bundle -> TRT runtime -> deserialize path.
//
// These tests build a syntactically valid .trtfb file with an invalid
// engine_plan payload, then call trtmc_create_pipeline repeatedly. This catches
// crashes in runtime/logger lifetime and validates that failures are reported
// through trtmc_last_error().
// =============================================================================

#include "cli/args.h"
#include "test_helpers.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_factory.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_pool.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef TRTMC_TEST_RUNTIME_LEGACY_INTERFACE_DSO
#error "TRTMC_TEST_RUNTIME_LEGACY_INTERFACE_DSO must name the current-ABI legacy fixture"
#endif
#ifndef TRTMC_TEST_RUNTIME_WRONG_VERSION_DSO
#error "TRTMC_TEST_RUNTIME_WRONG_VERSION_DSO must name the wrong-version fixture"
#endif
#ifndef TRTMC_TEST_RUNTIME_CAPTURING_DSO
#error "TRTMC_TEST_RUNTIME_CAPTURING_DSO must name the capturing fixture"
#endif

namespace {

int failures = 0;

#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(trtmc::LoadOptions) == 184, "LoadOptions legacy LP64 ABI layout changed");
static_assert(alignof(trtmc::LoadOptions) == 8, "LoadOptions legacy LP64 ABI alignment changed");
static_assert(sizeof(trtmc::PipelineContext) == 80,
              "PipelineContext legacy LP64 ABI layout changed");
static_assert(alignof(trtmc::PipelineContext) == 8,
              "PipelineContext legacy LP64 ABI alignment changed");
static_assert(sizeof(TrtmcPipelineOptions) == 40,
              "TrtmcPipelineOptions legacy LP64 ABI layout changed");
static_assert(alignof(TrtmcPipelineOptions) == 8,
              "TrtmcPipelineOptions legacy LP64 ABI alignment changed");
#endif

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

trtmc::cli::CliArgs parse_cli(std::initializer_list<std::string> arguments) {
    std::vector<std::string> storage(arguments);
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& argument : storage)
        argv.push_back(argument.data());
    return trtmc::cli::parse_args(static_cast<int>(argv.size()), argv.data());
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
                                     const std::string& model_id,
                                     const std::string& runtime_memory = "") {
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

    const std::string runtime_memory_field =
        runtime_memory.empty() ? std::string{} : "  \"runtime_memory\": " + runtime_memory + ",\n";

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
)" + runtime_memory_field +
           R"(  "sections": {
)" + sections_json +
           R"(
  }
})";
}

void write_bundle_with_sections(const std::filesystem::path& path,
                                const std::vector<BundleSectionSpec>& sections,
                                const std::string& model_id = "runtime-regression-test",
                                const std::string& runtime_memory = "") {
    // Internal .trtfb magic: "TRTFB\0\1\0"
    static constexpr unsigned char kBundleMagic[8] = {'T', 'R', 'T', 'F', 'B', '\0', '\x01', '\0'};

    const std::string header = build_bundle_header_json(sections, model_id, runtime_memory);
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

std::string runtime_memory_fixture_contract() {
    return R"({
    "contract_version": 2,
    "qualified_model_id": "qwen",
    "qualified_model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "qualified_config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "runtime_config_sha256": "a8348c09e4b6ab8c8df3b22aaf84b1e8ab0ee33ccc6a07aa5845286f5982b1df",
    "qualified_target": "gb300-trt-11.2",
    "qualified_runtime_stack": {"sm":"sm103","tensorrt":"11.2.0.113","cuda_runtime":"13.3","cudnn_backend":"9.20.0","cudnn_frontend_revision":"7b9b711c22b6823e87150213ecd8449260db8610","nvrtc":"13.3","driver":"580.105.08"},
    "native_kv_plugin_abi": 2,
    "model_context_limit": 32,
    "prefill_chunk_limit": 16,
    "kv_layout": "contiguous_runtime_v1",
    "kv_dtype": "float16",
    "kv_bytes_per_token": 256,
    "active_kv_profile_limits": [16, 32],
    "runtime_owned": true,
    "module_residency_calibration": {
      "schema_version": 1,
      "measurement_kind": "nvml_process_cumulative_first_use",
      "cuda_module_loading_mode": "lazy",
      "qualified_runtime_stack_sha256": "1b94f56092107d0fa1c6d43e2e8e4245c904ddc5967646f96ff8e64496e0f210",
      "plan_set_sha256": "60d8179e21e2b671155f90dfa50688046d06a11813284d1ef3c2d78d97a7ab9c",
      "evidence_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "plans": [
        {
          "section_name": "engine_plan",
          "section_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
          "role": "decode",
          "optimization_profile_count": 2
        },
        {
          "section_name": "prefill_engine_plan",
          "section_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
          "role": "prefill",
          "optimization_profile_count": 1
        }
      ],
      "profile_reserves": [
        {"covering_profile_limit": 16, "cumulative_reserve_bytes": 268435456},
        {"covering_profile_limit": 32, "cumulative_reserve_bytes": 536870912}
      ]
    }
  })";
}

std::string runtime_memory_fixture_config() {
    return R"({
  "runtime_strategy": "qwen_decoder_kv_cache",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
}

void write_runtime_memory_fixture_bundle(const std::filesystem::path& path) {
    const std::string config = runtime_memory_fixture_config();
    write_bundle_with_sections(path, {BundleSectionSpec{"config.json", config}}, "qwen",
                               runtime_memory_fixture_contract());
}

template <typename Function>
Function require_fixture_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr)
        throw std::runtime_error(std::string("missing runtime-memory fixture symbol ") + name);
    return reinterpret_cast<Function>(symbol);
}

struct RuntimeMemoryFixtureApi {
    using ResetFn = void (*)();
    using CountFn = std::uint32_t (*)();
    using U32Fn = std::uint32_t (*)();
    using U64Fn = std::uint64_t (*)();
    using DoubleFn = double (*)();

    void* handle{nullptr};
    ResetFn reset{nullptr};
    CountFn legacy_create_calls{nullptr};
    CountFn runtime_create_calls{nullptr};
    U32Fn captured_policy{nullptr};
    DoubleFn captured_fraction{nullptr};
    U64Fn captured_bytes{nullptr};
    U64Fn captured_context_kv_cache_size_bytes{nullptr};
    U64Fn captured_max_sequence_length{nullptr};
    U32Fn captured_max_sequence_length_explicit{nullptr};
};

RuntimeMemoryFixtureApi open_runtime_memory_fixture(const std::filesystem::path& dso) {
    RuntimeMemoryFixtureApi api;
    api.handle = dlopen(dso.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (api.handle == nullptr)
        throw std::runtime_error(std::string("could not open runtime-memory fixture: ") +
                                 dlerror());
    api.reset = require_fixture_symbol<RuntimeMemoryFixtureApi::ResetFn>(
        api.handle, "trtmc_test_runtime_plugin_reset");
    api.legacy_create_calls = require_fixture_symbol<RuntimeMemoryFixtureApi::CountFn>(
        api.handle, "trtmc_test_runtime_plugin_legacy_create_calls");
    api.runtime_create_calls = require_fixture_symbol<RuntimeMemoryFixtureApi::CountFn>(
        api.handle, "trtmc_test_runtime_plugin_runtime_create_calls");
    api.captured_policy = require_fixture_symbol<RuntimeMemoryFixtureApi::U32Fn>(
        api.handle, "trtmc_test_runtime_plugin_captured_policy");
    api.captured_fraction = require_fixture_symbol<RuntimeMemoryFixtureApi::DoubleFn>(
        api.handle, "trtmc_test_runtime_plugin_captured_fraction");
    api.captured_bytes = require_fixture_symbol<RuntimeMemoryFixtureApi::U64Fn>(
        api.handle, "trtmc_test_runtime_plugin_captured_bytes");
    api.captured_context_kv_cache_size_bytes =
        require_fixture_symbol<RuntimeMemoryFixtureApi::U64Fn>(
            api.handle, "trtmc_test_runtime_plugin_captured_context_kv_cache_size_bytes");
    api.captured_max_sequence_length = require_fixture_symbol<RuntimeMemoryFixtureApi::U64Fn>(
        api.handle, "trtmc_test_runtime_plugin_captured_max_sequence_length");
    api.captured_max_sequence_length_explicit =
        require_fixture_symbol<RuntimeMemoryFixtureApi::U32Fn>(
            api.handle, "trtmc_test_runtime_plugin_captured_max_sequence_length_explicit");
    api.reset();
    return api;
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

void expect_v2_creation_rejected(const std::filesystem::path& bundle_path,
                                 const TrtmcPipelineOptionsV2* options,
                                 const std::string& expected_message, const char* test_name) {
    auto* pipeline = trtmc_create_pipeline_v2(bundle_path.string().c_str(), options);
    check(pipeline == nullptr, test_name);
    const char* error = trtmc_last_error();
    check(error != nullptr && std::strlen(error) > 0, "V2 rejection sets trtmc_last_error");
    if (error != nullptr) {
        check(std::string(error).find(expected_message) != std::string::npos,
              "V2 rejection reports the failed contract");
    }
}

void test_pipeline_options_v2_reject_invalid_policy_values() {
    trtmc_test::TempDirGuard dir;
    const auto bundle_path = std::filesystem::path(dir.path()) / "v2_policy_validation.trtfb";
    const std::string config = R"({
  "runtime_strategy": "future_unknown_strategy",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(bundle_path, {BundleSectionSpec{"config.json", config}});

    expect_v2_creation_rejected(bundle_path, nullptr, "must not be null",
                                "null V2 options are rejected");

    TrtmcPipelineOptionsV2 options;
    trtmc_pipeline_options_v2_init(&options);
    options.struct_size = sizeof(options.struct_size);
    expect_v2_creation_rejected(bundle_path, &options, "struct_size is too small",
                                "short V2 struct is rejected");

    trtmc_pipeline_options_v2_init(&options);
    options.api_version = TRTMC_PIPELINE_OPTIONS_V2_API_VERSION + 1U;
    expect_v2_creation_rejected(bundle_path, &options, "api_version",
                                "unknown V2 API version is rejected");

    trtmc_pipeline_options_v2_init(&options);
    options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_AUTO;
    options.kv_cache_memory_bytes = 1;
    expect_v2_creation_rejected(bundle_path, &options, "conflicts",
                                "auto policy with byte value is rejected");

    for (const double fraction : {0.0, -0.1, 1.01, std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::infinity()}) {
        trtmc_pipeline_options_v2_init(&options);
        options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_FRACTION;
        options.kv_cache_memory_fraction = fraction;
        expect_v2_creation_rejected(bundle_path, &options, "fraction in (0, 1]",
                                    "invalid V2 fraction is rejected");
    }

    trtmc_pipeline_options_v2_init(&options);
    options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_FRACTION;
    options.kv_cache_memory_fraction = 0.8;
    options.kv_cache_memory_bytes = 1;
    expect_v2_creation_rejected(bundle_path, &options, "zero bytes",
                                "fraction and byte values are mutually exclusive");

    trtmc_pipeline_options_v2_init(&options);
    options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_BYTES;
    expect_v2_creation_rejected(bundle_path, &options, "positive bytes",
                                "zero-byte V2 policy is rejected");

    trtmc_pipeline_options_v2_init(&options);
    options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_BYTES;
    options.kv_cache_memory_bytes = 4096;
    options.kv_cache_memory_fraction = 0.5;
    expect_v2_creation_rejected(bundle_path, &options, "zero fraction",
                                "byte and fraction values are mutually exclusive");

    trtmc_pipeline_options_v2_init(&options);
    options.kv_cache_memory_policy = 99;
    expect_v2_creation_rejected(bundle_path, &options, "Unknown KV cache memory policy",
                                "unknown V2 KV policy is rejected");
}

void test_invalid_plan_bundle_reports_error() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path bundle_path =
        std::filesystem::path(dir.path()) / "invalid_engine_plan.trtfb";
    write_invalid_engine_bundle(bundle_path);
    expect_invalid_bundle_creation_fails(bundle_path.string(),
                                         "invalid plan bundle returns nullptr");
}

void test_missing_engine_plan_bundle_reports_error() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path bundle_path =
        std::filesystem::path(dir.path()) / "missing_engine_plan.trtfb";

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
        std::filesystem::path(dir.path()) / "unknown_strategy_missing_engine_plan.trtfb";

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

void test_runtime_kv_policy_rejects_unsupported_and_static_bundles() {
    trtmc_test::TempDirGuard dir;
    const auto root = std::filesystem::path(dir.path());

    const auto unsupported_path = root / "unsupported_dynamic_kv_policy.trtfb";
    const std::string unsupported_config = R"({
  "runtime_strategy": "future_unknown_strategy",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(unsupported_path,
                               {BundleSectionSpec{"config.json", unsupported_config}});

    trtmc::LoadOptionsV2 percentage_options;
    percentage_options.kv_cache_memory_policy = trtmc::KvCacheMemoryPolicy::kFraction;
    percentage_options.kv_cache_memory_fraction = 0.8;
    bool unsupported_rejected = false;
    try {
        (void)trtmc::load(unsupported_path.string(), percentage_options);
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        unsupported_rejected = message.find("does not declare runtime_memory contract version 2") !=
                                   std::string::npos &&
                               message.find("engine_plan") == std::string::npos;
    }
    check(unsupported_rejected, "unsupported strategy rejects percentage KV policy");

    TrtmcPipelineOptionsV2 v2_options;
    trtmc_pipeline_options_v2_init(&v2_options);
    v2_options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_FRACTION;
    v2_options.kv_cache_memory_fraction = 0.8;
    expect_v2_creation_rejected(unsupported_path, &v2_options,
                                "does not declare runtime_memory contract version 2",
                                "C ABI V2 rejects percentage policy on unsupported bundle");
    const std::string v2_fraction_error = trtmc_last_error();
    check(v2_fraction_error.find("engine_plan") == std::string::npos,
          "C ABI V2 percentage policy rejects before engine deserialization");

    trtmc_pipeline_options_v2_init(&v2_options);
    v2_options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_BYTES;
    v2_options.kv_cache_memory_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    expect_v2_creation_rejected(unsupported_path, &v2_options,
                                "does not declare runtime_memory contract version 2",
                                "C ABI V2 rejects byte policy on unsupported bundle");

    trtmc_pipeline_options_v2_init(&v2_options);
    v2_options.max_sequence_length = 32768;
    v2_options.max_sequence_length_explicit = 1;
    expect_v2_creation_rejected(unsupported_path, &v2_options,
                                "does not declare runtime_memory contract version 2",
                                "C ABI V2 rejects sequence policy on unsupported bundle");

    const auto static_qwen_path = root / "static_qwen_auto_policy.trtfb";
    const std::string static_qwen_config = R"({
  "runtime_strategy": "qwen_decoder_kv_cache",
  "dynamic_kv_cache": false,
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(static_qwen_path,
                               {BundleSectionSpec{"config.json", static_qwen_config}}, "qwen");

    trtmc::LoadOptionsV2 explicit_auto;
    explicit_auto.kv_cache_memory_policy = trtmc::KvCacheMemoryPolicy::kAuto;
    bool static_rejected = false;
    try {
        (void)trtmc::load(static_qwen_path.string(), explicit_auto);
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        static_rejected = message.find("does not declare runtime_memory contract version 2") !=
                              std::string::npos &&
                          message.find("engine_plan") == std::string::npos;
    }
    check(static_rejected, "static Qwen bundle rejects explicit auto KV policy");

    const auto static_rtx_path = root / "static_rtx_auto_policy.trtfb";
    const std::string static_rtx_config = R"({
  "runtime_strategy": "qwen_decoder_kv_cache",
  "engine_backend": "trt_rtx",
  "dynamic_kv_cache": false,
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(static_rtx_path,
                               {BundleSectionSpec{"config.json", static_rtx_config}}, "qwen");
    bool static_rtx_rejected = false;
    try {
        (void)trtmc::load(static_rtx_path.string(), percentage_options);
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        static_rtx_rejected =
            message.find("does not declare runtime_memory contract version 2") !=
                std::string::npos &&
            message.find("trt_rtx") == std::string::npos &&
            message.find("backend") == std::string::npos;
    }
    check(static_rtx_rejected,
          "static TensorRT-RTX bundle rejects runtime-memory policy before backend dispatch");

    const auto dynamic_qwen_path = root / "dynamic_qwen_pool.trtfb";
    const std::string dynamic_qwen_config = R"({
  "runtime_strategy": "qwen_decoder_kv_cache",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(dynamic_qwen_path,
                               {BundleSectionSpec{"config.json", dynamic_qwen_config}}, "qwen",
                               runtime_memory_fixture_contract());
    bool pool_rejected = false;
    try {
        (void)trtmc::PipelineFactory::from_bundle_pool(dynamic_qwen_path.string(), 2);
    } catch (const std::invalid_argument& error) {
        pool_rejected =
            std::string(error.what()).find("does not yet support runtime-sized KV cache bundles") !=
            std::string::npos;
    }
    check(pool_rejected, "dynamic KV beta rejects ambiguous per-lane pool budgeting");
}

void test_runtime_memory_config_integrity_precedes_dispatch() {
    trtmc_test::TempDirGuard dir;
    const auto root = std::filesystem::path(dir.path());
    const std::string contract = runtime_memory_fixture_contract();

    const auto missing_path = root / "dynamic_missing_config.trtfb";
    write_bundle_with_sections(missing_path, {}, "qwen", contract);

    const auto empty_path = root / "dynamic_empty_config.trtfb";
    write_bundle_with_sections(empty_path, {BundleSectionSpec{"config.json", ""}},
                               "qwen", contract);

    const auto tampered_path = root / "dynamic_tampered_config.trtfb";
    write_bundle_with_sections(
        tampered_path,
        {BundleSectionSpec{"config.json", runtime_memory_fixture_config() + " "}},
        "qwen", contract);

    const auto expect_pre_dispatch_rejection =
        [&](const std::filesystem::path& path, const std::string& expected,
            const char* test_name) {
            trtmc::LoadOptionsV2 options;
            options.model_plugin_search_paths = {(root / "unused-model-plugins").string()};
            options.backend_search_paths = {(root / "unused-backends").string()};
            std::string message;
            try {
                (void)trtmc::load(path.string(), options);
            } catch (const std::runtime_error& error) {
                message = error.what();
            }
            const bool rejected =
                message.find(expected) != std::string::npos &&
                message.find("model plugin") == std::string::npos &&
                message.find("Backend") == std::string::npos &&
                message.find("engine") == std::string::npos;
            check(rejected, test_name);
        };

    expect_pre_dispatch_rejection(
        missing_path, "missing a non-empty config.json",
        "runtime-memory bundle without config fails before dispatch");
    expect_pre_dispatch_rejection(
        empty_path, "missing a non-empty config.json",
        "runtime-memory bundle with empty config fails before dispatch");
    expect_pre_dispatch_rejection(
        tampered_path, "config.json hash mismatch",
        "runtime-memory bundle with tampered config fails before dispatch");
}

void test_cpp_v2_options_fail_before_bundle_io() {
    trtmc_test::TempDirGuard dir;
    const auto root = std::filesystem::path(dir.path());

    trtmc::LoadOptionsV2 short_options;
    short_options.struct_size = sizeof(short_options.struct_size);
    bool short_rejected = false;
    try {
        (void)trtmc::load((root / "does-not-need-to-exist.trtfb").string(), short_options);
    } catch (const std::invalid_argument& error) {
        short_rejected =
            std::string(error.what()).find("struct_size is too small") != std::string::npos;
    }
    check(short_rejected, "C++ LoadOptionsV2 short struct fails before bundle I/O");

    trtmc::LoadOptionsV2 wrong_api;
    wrong_api.api_version = trtmc::kLoadOptionsV2ApiVersion + 1U;
    bool api_rejected = false;
    try {
        (void)trtmc::load((root / "does-not-need-to-exist.trtfb").string(), wrong_api);
    } catch (const std::invalid_argument& error) {
        api_rejected = std::string(error.what()).find("api_version") != std::string::npos;
    }
    check(api_rejected, "C++ LoadOptionsV2 unknown API fails before bundle I/O");
}

void test_current_abi_runtime_plugin_interface_rejection(const std::filesystem::path& dso,
                                                         bool wrong_version) {
    setenv("TRTMC_MODEL_PLUGIN_STRICT", "1", 1);
    unsetenv("TRTMC_MODEL_PLUGIN_DIR");

    trtmc_test::TempDirGuard dir;
    const auto bundle_path = std::filesystem::path(dir.path()) / "runtime-plugin-interface.trtfb";
    write_runtime_memory_fixture_bundle(bundle_path);
    auto fixture = open_runtime_memory_fixture(dso);

    trtmc::LoadOptionsV2 options;
    options.model_plugin_search_paths = {dso.parent_path().string()};
    bool rejected = false;
    try {
        (void)trtmc::load(bundle_path.string(), options);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        const std::string expected = wrong_version
                                         ? "incompatible model plugin runtime-memory API version"
                                         : "requires model plugin runtime-memory interface V1";
        rejected = message.find(expected) != std::string::npos &&
                   message.find("backend") == std::string::npos;
    }
    check(rejected, wrong_version
                        ? "current-ABI DSO rejects wrong runtime-memory plugin API version"
                        : "current-ABI DSO rejects missing runtime-memory plugin interface");
    check(fixture.legacy_create_calls() == 0 && fixture.runtime_create_calls() == 0,
          "runtime-memory interface rejection precedes both model create entry points");

    dlclose(fixture.handle);
    unsetenv("TRTMC_MODEL_PLUGIN_STRICT");
}

void test_legacy_cpp_and_c_surfaces_select_dynamic_implicit_auto(const std::filesystem::path& dso) {
    setenv("TRTMC_MODEL_PLUGIN_STRICT", "1", 1);
    unsetenv("TRTMC_MODEL_PLUGIN_DIR");

    trtmc_test::TempDirGuard dir;
    const auto bundle_path = std::filesystem::path(dir.path()) / "legacy-implicit-auto.trtfb";
    write_runtime_memory_fixture_bundle(bundle_path);
    auto fixture = open_runtime_memory_fixture(dso);

    // Deliberately initialize every field in the original aggregate before
    // selecting the independently built current-ABI fixture.
    trtmc::LoadOptions legacy_aggregate{"", "", false, 0, "", {}, {}, {}};
    legacy_aggregate.model_plugin_search_paths = {dso.parent_path().string()};
    auto cpp_pipeline = trtmc::load(bundle_path.string(), legacy_aggregate);
    check(cpp_pipeline != nullptr, "legacy aggregate LoadOptions loads dynamic bundle");
    check(fixture.legacy_create_calls() == 0 && fixture.runtime_create_calls() == 1,
          "legacy aggregate dispatches through runtime-memory plugin interface");
    check(fixture.captured_policy() ==
                  static_cast<std::uint32_t>(trtmc::KvCacheMemoryPolicy::kAuto) &&
              fixture.captured_fraction() == 0.90 && fixture.captured_bytes() == 0,
          "legacy aggregate receives implicit 90 percent auto policy");

    fixture.reset();
    TrtmcPipelineOptions old_c_options{};
    auto* c_pipeline = trtmc_create_pipeline_ex(bundle_path.string().c_str(), &old_c_options);
    check(c_pipeline != nullptr, "legacy trtmc_create_pipeline_ex loads dynamic bundle");
    check(fixture.legacy_create_calls() == 0 && fixture.runtime_create_calls() == 1,
          "legacy C create_ex dispatches through runtime-memory plugin interface");
    check(fixture.captured_policy() ==
                  static_cast<std::uint32_t>(trtmc::KvCacheMemoryPolicy::kAuto) &&
              fixture.captured_fraction() == 0.90 && fixture.captured_bytes() == 0 &&
              fixture.captured_max_sequence_length() == 0 &&
              fixture.captured_max_sequence_length_explicit() == 0,
          "legacy C create_ex receives implicit 90 percent auto policy");
    delete c_pipeline;

    dlclose(fixture.handle);
    unsetenv("TRTMC_MODEL_PLUGIN_STRICT");
}

void test_cli_legacy_size_and_runtime_memory_policy_are_distinct(const std::filesystem::path& dso) {
    setenv("TRTMC_MODEL_PLUGIN_STRICT", "1", 1);
    unsetenv("TRTMC_MODEL_PLUGIN_DIR");

    trtmc_test::TempDirGuard dir;
    const auto root = std::filesystem::path(dir.path());
    const auto static_bundle_path = root / "cli-static-legacy-size.trtfb";
    const auto dynamic_bundle_path = root / "cli-dynamic-runtime-memory.trtfb";
    const std::string config = R"({
  "runtime_strategy": "qwen_decoder_kv_cache",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    write_bundle_with_sections(static_bundle_path, {BundleSectionSpec{"config.json", config}},
                               "qwen");
    write_bundle_with_sections(dynamic_bundle_path, {BundleSectionSpec{"config.json", config}},
                               "qwen", runtime_memory_fixture_contract());
    auto fixture = open_runtime_memory_fixture(dso);
    const std::string plugin_dir = dso.parent_path().string();

    const auto legacy_args =
        parse_cli({"trtmc", "run", static_bundle_path.string(), "--prompt", "hello",
                   "--kv-cache-size", "4GiB", "--model-plugin-dir", plugin_dir});
    check(!legacy_args.parse_error, "legacy static-bundle CLI alias parses");
    check(legacy_args.kv_cache_size_explicitly_set && !legacy_args.kv_cache_memory.explicitly_set,
          "legacy CLI alias remains distinct from runtime policy");
    const auto legacy_options = trtmc::cli::make_load_options(legacy_args);
    check(legacy_options.kv_cache_size_bytes == 4ULL * 1024ULL * 1024ULL * 1024ULL &&
              legacy_options.kv_cache_memory_policy == trtmc::KvCacheMemoryPolicy::kUnspecified &&
              legacy_options.kv_cache_memory_bytes == 0,
          "legacy CLI alias maps only to LoadOptionsV2 compatibility bytes");
    auto static_pipeline = trtmc::load(static_bundle_path.string(), legacy_options);
    check(static_pipeline != nullptr, "legacy CLI alias loads a static bundle");
    check(fixture.legacy_create_calls() == 1 && fixture.runtime_create_calls() == 0 &&
              fixture.captured_context_kv_cache_size_bytes() == 4ULL * 1024ULL * 1024ULL * 1024ULL,
          "static pipeline receives legacy CLI byte budget through PipelineContext");

    fixture.reset();
    const auto dynamic_args =
        parse_cli({"trtmc", "run", dynamic_bundle_path.string(), "--prompt", "hello",
                   "--kv-cache-memory", "3GiB", "--model-plugin-dir", plugin_dir});
    check(!dynamic_args.parse_error, "canonical dynamic-memory CLI option parses");
    check(dynamic_args.kv_cache_memory.explicitly_set && !dynamic_args.kv_cache_size_explicitly_set,
          "canonical dynamic-memory CLI option does not select legacy alias");
    const auto dynamic_options = trtmc::cli::make_load_options(dynamic_args);
    check(dynamic_options.kv_cache_size_bytes == 0 &&
              dynamic_options.kv_cache_memory_policy == trtmc::KvCacheMemoryPolicy::kBytes &&
              dynamic_options.kv_cache_memory_bytes == 3ULL * 1024ULL * 1024ULL * 1024ULL,
          "canonical dynamic-memory spelling maps only to the new policy");
    auto dynamic_pipeline = trtmc::load(dynamic_bundle_path.string(), dynamic_options);
    check(dynamic_pipeline != nullptr, "canonical policy loads a runtime-memory bundle");
    check(fixture.legacy_create_calls() == 0 && fixture.runtime_create_calls() == 1 &&
              fixture.captured_policy() ==
                  static_cast<std::uint32_t>(trtmc::KvCacheMemoryPolicy::kBytes) &&
              fixture.captured_bytes() == 3ULL * 1024ULL * 1024ULL * 1024ULL &&
              fixture.captured_context_kv_cache_size_bytes() == 0,
          "dynamic pipeline receives canonical byte policy outside legacy context");

    fixture.reset();
    const auto invalid_static_args =
        parse_cli({"trtmc", "run", static_bundle_path.string(), "--prompt", "hello",
                   "--kv-cache-memory", "3GiB", "--model-plugin-dir", plugin_dir});
    bool invalid_static_rejected = false;
    try {
        (void)trtmc::load(static_bundle_path.string(),
                          trtmc::cli::make_load_options(invalid_static_args));
    } catch (const std::invalid_argument& error) {
        invalid_static_rejected =
            std::string(error.what()).find("does not declare runtime_memory contract version 2") !=
            std::string::npos;
    }
    check(invalid_static_rejected,
          "canonical runtime-memory policy fails closed on a static bundle");
    check(fixture.legacy_create_calls() == 0 && fixture.runtime_create_calls() == 0,
          "static canonical-policy rejection precedes model plugin creation");

    dlclose(fixture.handle);
    unsetenv("TRTMC_MODEL_PLUGIN_STRICT");
}

void test_invalid_plan_bundle_repeatable() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path bundle_path =
        std::filesystem::path(dir.path()) / "invalid_engine_plan_loop.trtfb";
    write_invalid_engine_bundle(bundle_path);

    for (int i = 0; i < 25; ++i) {
        expect_invalid_bundle_creation_fails(bundle_path.string(),
                                             "invalid plan bundle repeated returns nullptr");
    }
}

void run_isolated_fixture_scenario(const std::string& scenario) {
    if (scenario == "--runtime-legacy-interface") {
        test_current_abi_runtime_plugin_interface_rejection(TRTMC_TEST_RUNTIME_LEGACY_INTERFACE_DSO,
                                                            false);
        return;
    }
    if (scenario == "--runtime-wrong-version") {
        test_current_abi_runtime_plugin_interface_rejection(TRTMC_TEST_RUNTIME_WRONG_VERSION_DSO,
                                                            true);
        return;
    }
    if (scenario == "--runtime-capturing") {
        test_legacy_cpp_and_c_surfaces_select_dynamic_implicit_auto(
            TRTMC_TEST_RUNTIME_CAPTURING_DSO);
        return;
    }
    if (scenario == "--cli-kv-alias") {
        test_cli_legacy_size_and_runtime_memory_policy_are_distinct(
            TRTMC_TEST_RUNTIME_CAPTURING_DSO);
        return;
    }
    throw std::invalid_argument("unknown isolated runtime-memory fixture scenario: " + scenario);
}

void expect_isolated_fixture_succeeds(const std::filesystem::path& executable, const char* scenario,
                                      const char* test_name) {
    const pid_t child = fork();
    if (child < 0) {
        check(false, test_name);
        return;
    }
    if (child == 0) {
        execl(executable.c_str(), executable.c_str(), scenario, nullptr);
        _exit(127);
    }
    int status = 0;
    const pid_t waited = waitpid(child, &status, 0);
    check(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, test_name);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        try {
            run_isolated_fixture_scenario(argv[1]);
        } catch (const std::exception& error) {
            std::cerr << "isolated runtime-memory fixture failed: " << error.what() << '\n';
            return 1;
        }
        return failures == 0 ? 0 : 1;
    }
    if (argc != 1) {
        std::cerr << "unexpected test arguments\n";
        return 1;
    }

    test_pipeline_options_v2_reject_invalid_policy_values();
    test_invalid_plan_bundle_reports_error();
    test_missing_engine_plan_bundle_reports_error();
    test_unknown_strategy_reports_new_runtime_unsupported_strategy_error();
    test_runtime_kv_policy_rejects_unsupported_and_static_bundles();
    test_runtime_memory_config_integrity_precedes_dispatch();
    test_cpp_v2_options_fail_before_bundle_io();
    const auto executable = std::filesystem::canonical(argv[0]);
    expect_isolated_fixture_succeeds(executable, "--runtime-legacy-interface",
                                     "current-ABI legacy interface fixture passes");
    expect_isolated_fixture_succeeds(executable, "--runtime-wrong-version",
                                     "current-ABI wrong-version fixture passes");
    expect_isolated_fixture_succeeds(executable, "--runtime-capturing",
                                     "current-ABI capturing fixture passes");
    expect_isolated_fixture_succeeds(executable, "--cli-kv-alias",
                                     "CLI legacy KV alias compatibility fixture passes");
    test_invalid_plan_bundle_repeatable();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI runtime regression tests passed.\n";
    return 0;
}
