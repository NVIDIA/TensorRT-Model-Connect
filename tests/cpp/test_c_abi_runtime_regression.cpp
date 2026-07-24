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

#include "test_helpers.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_factory.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_pool.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

class LegacyOnlyPlugin final : public trtmc::IPipelinePlugin {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext&) override {
        create_called = true;
        return nullptr;
    }

    bool create_called{false};
};

class WrongRuntimeMemoryVersionPlugin final : public trtmc::IPipelinePlugin,
                                              public trtmc::IRuntimeMemoryPipelinePluginV1 {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext&) override {
        create_called = true;
        return nullptr;
    }

    std::uint32_t runtime_memory_plugin_api_version() const override {
        return trtmc::kRuntimeMemoryPluginApiVersionV1 + 1U;
    }

    std::unique_ptr<trtmc::IPipeline>
    create_runtime_memory(const trtmc::PipelineContext&,
                          const trtmc::RuntimeMemoryPluginOptionsV1&) override {
        runtime_create_called = true;
        return nullptr;
    }

    bool create_called{false};
    bool runtime_create_called{false};
};

class CapturingPipeline final : public trtmc::IPipeline {
  public:
    const char* model_id() const override { return "runtime-memory-capture"; }
    const char* pipeline_type() const override { return "runtime-memory-capture"; }
};

class CapturingRuntimeMemoryPlugin final : public trtmc::IPipelinePlugin,
                                           public trtmc::IRuntimeMemoryPipelinePluginV1 {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext&) override {
        legacy_create_called = true;
        return std::make_unique<CapturingPipeline>();
    }

    std::unique_ptr<trtmc::IPipeline>
    create_runtime_memory(const trtmc::PipelineContext&,
                          const trtmc::RuntimeMemoryPluginOptionsV1& options) override {
        ++runtime_create_calls;
        captured_options = options;
        return std::make_unique<CapturingPipeline>();
    }

    void reset_capture() {
        legacy_create_called = false;
        runtime_create_calls = 0;
        captured_options = {};
    }

    bool legacy_create_called{false};
    int runtime_create_calls{0};
    trtmc::RuntimeMemoryPluginOptionsV1 captured_options;
};

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
        unsupported_rejected = message.find("does not declare runtime_memory contract version 1") !=
                                   std::string::npos &&
                               message.find("engine_plan") == std::string::npos;
    }
    check(unsupported_rejected, "unsupported strategy rejects percentage KV policy");

    TrtmcPipelineOptionsV2 v2_options;
    trtmc_pipeline_options_v2_init(&v2_options);
    v2_options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_FRACTION;
    v2_options.kv_cache_memory_fraction = 0.8;
    expect_v2_creation_rejected(unsupported_path, &v2_options,
                                "does not declare runtime_memory contract version 1",
                                "C ABI V2 rejects percentage policy on unsupported bundle");
    const std::string v2_fraction_error = trtmc_last_error();
    check(v2_fraction_error.find("engine_plan") == std::string::npos,
          "C ABI V2 percentage policy rejects before engine deserialization");

    trtmc_pipeline_options_v2_init(&v2_options);
    v2_options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_BYTES;
    v2_options.kv_cache_memory_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    expect_v2_creation_rejected(unsupported_path, &v2_options,
                                "does not declare runtime_memory contract version 1",
                                "C ABI V2 rejects byte policy on unsupported bundle");

    trtmc_pipeline_options_v2_init(&v2_options);
    v2_options.max_sequence_length = 32768;
    v2_options.max_sequence_length_explicit = 1;
    expect_v2_creation_rejected(unsupported_path, &v2_options,
                                "does not declare runtime_memory contract version 1",
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
        static_rejected = message.find("does not declare runtime_memory contract version 1") !=
                              std::string::npos &&
                          message.find("engine_plan") == std::string::npos;
    }
    check(static_rejected, "static Qwen bundle rejects explicit auto KV policy");

    const auto dynamic_qwen_path = root / "dynamic_qwen_pool.trtfb";
    const std::string dynamic_qwen_config = R"({
  "runtime_strategy": "qwen_decoder_kv_cache",
  "hidden_size": 64,
  "num_attention_heads": 1,
  "num_key_value_heads": 1
})";
    const std::string runtime_memory = R"({
    "contract_version": 1,
    "qualified_model_id": "qwen",
    "qualified_model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "qualified_config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "qualified_target": "gb300-trt-11.2",
    "qualified_runtime_stack": {"sm":"sm103","tensorrt":"11.2.0.113","cuda_runtime":"13.3","cudnn_backend":"9.20.0","cudnn_frontend_revision":"7b9b711c22b6823e87150213ecd8449260db8610","nvrtc":"13.3","driver":"580.105.08"},
    "native_kv_plugin_abi": 2,
    "model_context_limit": 32,
    "prefill_chunk_limit": 16,
    "kv_layout": "layer_major_contiguous_k_then_v",
    "kv_dtype": "float16",
    "kv_bytes_per_token": 256,
    "active_kv_profile_limits": [16, 32],
    "runtime_owned": true
  })";
    write_bundle_with_sections(dynamic_qwen_path,
                               {BundleSectionSpec{"config.json", dynamic_qwen_config}}, "qwen",
                               runtime_memory);
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

void test_cpp_v2_and_mixed_plugin_versions_fail_before_backend_load() {
    trtmc_test::TempDirGuard dir;
    const auto root = std::filesystem::path(dir.path());
    const std::string runtime_memory = R"({
    "contract_version": 1,
    "qualified_model_id": "mixed-version-test",
    "qualified_model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "qualified_config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "qualified_target": "gb300-trt-11.2",
    "qualified_runtime_stack": {"sm":"sm103","tensorrt":"11.2.0.113","cuda_runtime":"13.3","cudnn_backend":"9.20.0","cudnn_frontend_revision":"7b9b711c22b6823e87150213ecd8449260db8610","nvrtc":"13.3","driver":"580.105.08"},
    "native_kv_plugin_abi": 2,
    "model_context_limit": 32,
    "prefill_chunk_limit": 16,
    "kv_layout": "contiguous_runtime_v1",
    "kv_dtype": "float16",
    "kv_bytes_per_token": 256,
    "active_kv_profile_limits": [16, 32],
    "runtime_owned": true
  })";

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

    LegacyOnlyPlugin legacy_plugin;
    constexpr const char* kLegacyStrategy = "test_runtime_memory_legacy_plugin";
    trtmc::PipelineRegistry::instance().register_plugin(kLegacyStrategy, &legacy_plugin);
    const auto legacy_bundle = root / "legacy-plugin.trtfb";
    const std::string legacy_config = std::string("{\"runtime_strategy\":\"") + kLegacyStrategy +
                                      "\",\"hidden_size\":64,\"num_attention_heads\":1,"
                                      "\"num_key_value_heads\":1}";
    write_bundle_with_sections(legacy_bundle, {BundleSectionSpec{"config.json", legacy_config}},
                               "mixed-version-test", runtime_memory);
    bool legacy_rejected = false;
    try {
        (void)trtmc::load(legacy_bundle.string(), trtmc::LoadOptionsV2{});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        legacy_rejected = message.find("requires model plugin runtime-memory interface V1") !=
                              std::string::npos &&
                          message.find("backend") == std::string::npos;
    }
    check(legacy_rejected, "new core rejects legacy plugin before backend deserialization");
    check(!legacy_plugin.create_called,
          "legacy plugin create is never invoked for runtime_memory bundle");

    WrongRuntimeMemoryVersionPlugin wrong_plugin;
    constexpr const char* kWrongStrategy = "test_runtime_memory_wrong_version_plugin";
    trtmc::PipelineRegistry::instance().register_plugin(kWrongStrategy, &wrong_plugin);
    const auto wrong_bundle = root / "wrong-plugin-version.trtfb";
    const std::string wrong_config = std::string("{\"runtime_strategy\":\"") + kWrongStrategy +
                                     "\",\"hidden_size\":64,\"num_attention_heads\":1,"
                                     "\"num_key_value_heads\":1}";
    write_bundle_with_sections(wrong_bundle, {BundleSectionSpec{"config.json", wrong_config}},
                               "mixed-version-test", runtime_memory);
    bool wrong_version_rejected = false;
    try {
        (void)trtmc::load(wrong_bundle.string(), trtmc::LoadOptionsV2{});
    } catch (const std::runtime_error& error) {
        wrong_version_rejected =
            std::string(error.what())
                .find("incompatible model plugin runtime-memory API version") != std::string::npos;
    }
    check(wrong_version_rejected, "new core rejects wrong runtime-memory plugin API version");
    check(!wrong_plugin.create_called && !wrong_plugin.runtime_create_called,
          "wrong-version plugin is rejected before either create entry point");
}

void test_legacy_cpp_and_c_surfaces_select_dynamic_implicit_auto() {
    trtmc_test::TempDirGuard dir;
    const auto bundle_path = std::filesystem::path(dir.path()) / "legacy-implicit-auto.trtfb";
    constexpr const char* kStrategy = "test_runtime_memory_legacy_implicit_auto";
    const std::string config = std::string("{\"runtime_strategy\":\"") + kStrategy +
                               "\",\"hidden_size\":64,\"num_attention_heads\":1,"
                               "\"num_key_value_heads\":1}";
    const std::string runtime_memory = R"({
    "contract_version": 1,
    "qualified_model_id": "runtime-memory-capture",
    "qualified_model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "qualified_config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "qualified_target": "gb300-trt-11.2",
    "qualified_runtime_stack": {"sm":"sm103","tensorrt":"11.2.0.113","cuda_runtime":"13.3","cudnn_backend":"9.20.0","cudnn_frontend_revision":"7b9b711c22b6823e87150213ecd8449260db8610","nvrtc":"13.3","driver":"580.105.08"},
    "native_kv_plugin_abi": 2,
    "model_context_limit": 32,
    "prefill_chunk_limit": 16,
    "kv_layout": "contiguous_runtime_v1",
    "kv_dtype": "float16",
    "kv_bytes_per_token": 256,
    "active_kv_profile_limits": [16, 32],
    "runtime_owned": true
  })";
    write_bundle_with_sections(bundle_path, {BundleSectionSpec{"config.json", config}},
                               "runtime-memory-capture", runtime_memory);

    static CapturingRuntimeMemoryPlugin plugin;
    plugin.reset_capture();
    trtmc::PipelineRegistry::instance().register_plugin(kStrategy, &plugin);

    // Deliberately use every field in the original aggregate. This is both a
    // source-compatibility guard and proof that the legacy C++ overload maps a
    // runtime-memory bundle to the implicit automatic policy.
    trtmc::LoadOptions legacy_aggregate{"", "", false, 0, "", {}, {}, {}};
    auto cpp_pipeline = trtmc::load(bundle_path.string(), legacy_aggregate);
    check(cpp_pipeline != nullptr, "legacy aggregate LoadOptions loads dynamic bundle");
    check(!plugin.legacy_create_called && plugin.runtime_create_calls == 1,
          "legacy aggregate dispatches through runtime-memory plugin interface");
    check(plugin.captured_options.kv_cache_memory_policy == trtmc::KvCacheMemoryPolicy::kAuto &&
              plugin.captured_options.kv_cache_memory_fraction == 0.90 &&
              plugin.captured_options.kv_cache_memory_bytes == 0,
          "legacy aggregate receives implicit 90 percent auto policy");

    plugin.reset_capture();
    TrtmcPipelineOptions old_c_options{};
    auto* c_pipeline = trtmc_create_pipeline_ex(bundle_path.string().c_str(), &old_c_options);
    check(c_pipeline != nullptr, "legacy trtmc_create_pipeline_ex loads dynamic bundle");
    check(!plugin.legacy_create_called && plugin.runtime_create_calls == 1,
          "legacy C create_ex dispatches through runtime-memory plugin interface");
    check(plugin.captured_options.kv_cache_memory_policy == trtmc::KvCacheMemoryPolicy::kAuto &&
              plugin.captured_options.kv_cache_memory_fraction == 0.90 &&
              plugin.captured_options.kv_cache_memory_bytes == 0 &&
              plugin.captured_options.max_sequence_length == 0 &&
              plugin.captured_options.max_sequence_length_explicit == 0,
          "legacy C create_ex receives implicit 90 percent auto policy");
    delete c_pipeline;
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

} // namespace

int main() {
    test_pipeline_options_v2_reject_invalid_policy_values();
    test_invalid_plan_bundle_reports_error();
    test_missing_engine_plan_bundle_reports_error();
    test_unknown_strategy_reports_new_runtime_unsupported_strategy_error();
    test_runtime_kv_policy_rejects_unsupported_and_static_bundles();
    test_cpp_v2_and_mixed_plugin_versions_fail_before_backend_load();
    test_legacy_cpp_and_c_surfaces_select_dynamic_implicit_auto();
    test_invalid_plan_bundle_repeatable();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All C ABI runtime regression tests passed.\n";
    return 0;
}
