/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/pipeline.h"

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef TRTMC_TEST_LEGACY_PLUGIN_DSO
#error "TRTMC_TEST_LEGACY_PLUGIN_DSO must name the independently built legacy plugin"
#endif
#ifndef TRTMC_TEST_ABI_BACKEND_DSO
#error "TRTMC_TEST_ABI_BACKEND_DSO must name the independently built test backend"
#endif

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class TempDirectory {
  public:
    TempDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("trtmc-abi-" + std::to_string(static_cast<long long>(getpid())));
        std::filesystem::create_directories(path_);
    }
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    const std::filesystem::path& path() const { return path_; }

  private:
    std::filesystem::path path_;
};

void write_u64_le(std::ofstream& output, std::uint64_t value) {
    unsigned char bytes[8];
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<unsigned char>((value >> (8 * index)) & 0xffU);
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

std::string header_json(const std::string& config, const std::string& runtime_memory) {
    const std::string runtime_memory_field =
        runtime_memory.empty() ? std::string{} : "\"runtime_memory\":" + runtime_memory + ",";
    return std::string("{") +
           "\"model_id\":\"qwen\","
           "\"model_type\":\"unit-test\","
           "\"family\":\"qwen\","
           "\"hidden_size\":64,"
           "\"num_layers\":1,"
           "\"num_attention_heads\":1,"
           "\"num_key_value_heads\":1,"
           "\"max_cache_length\":32," +
           runtime_memory_field +
           "\"sections\":{\"config.json\":{\"offset\":0,\"size\":" + std::to_string(config.size()) +
           "}}}";
}

void write_bundle(const std::filesystem::path& path, const std::string& runtime_memory = "") {
    static constexpr unsigned char magic[8] = {'T', 'R', 'T', 'F', 'B', '\0', '\x01', '\0'};
    const std::string config =
        R"({"runtime_strategy":"qwen_decoder_kv_cache","engine_backend":"abi_fixture","hidden_size":64,"num_attention_heads":1,"num_key_value_heads":1})";
    const std::string header = header_json(config, runtime_memory);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    write_u64_le(output, header.size());
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(config.data(), static_cast<std::streamsize>(config.size()));
}

using CountFn = std::uint32_t (*)();
using ResetFn = void (*)();

void* require_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr)
        throw std::runtime_error(std::string("missing fixture symbol ") + name);
    return symbol;
}

} // namespace

int main() {
    // This executable must exercise only the independently compiled stale
    // fixture.  Without strict loading the production search fallbacks can
    // continue after rejecting that DSO and load the current in-tree Qwen
    // plugin, turning the negative ABI test into a false success path.
    setenv("TRTMC_MODEL_PLUGIN_STRICT", "1", 1);
    unsetenv("TRTMC_MODEL_PLUGIN_DIR");

    TempDirectory temp;
    const auto dynamic_bundle = temp.path() / "dynamic.trtfb";
    const auto static_bundle = temp.path() / "static.trtfb";
    const std::string runtime_memory = R"({
      "contract_version":1,
      "qualified_model_id":"qwen",
      "qualified_model_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "qualified_config_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "qualified_target":"gb300-trt-11.2",
      "qualified_runtime_stack":{"sm":"sm103","tensorrt":"11.2.0.113","cuda_runtime":"13.3","cudnn_backend":"9.20.0","cudnn_frontend_revision":"7b9b711c22b6823e87150213ecd8449260db8610","nvrtc":"13.3","driver":"580.105.08"},
      "native_kv_plugin_abi":2,
      "model_context_limit":32,
      "prefill_chunk_limit":16,
      "kv_layout":"contiguous_runtime_v1",
      "kv_dtype":"float16",
      "kv_bytes_per_token":256,
      "active_kv_profile_limits":[16,32],
      "runtime_owned":true
    })";
    write_bundle(dynamic_bundle, runtime_memory);
    write_bundle(static_bundle);

    const std::filesystem::path plugin_dso = TRTMC_TEST_LEGACY_PLUGIN_DSO;
    const std::filesystem::path backend_dso = TRTMC_TEST_ABI_BACKEND_DSO;

    trtmc::LoadOptionsV2 options;
    options.model_plugin_search_paths = {plugin_dso.parent_path().string()};
    options.backend_search_paths = {backend_dso.parent_path().string()};

    void* plugin_handle = dlopen(plugin_dso.c_str(), RTLD_NOW | RTLD_LOCAL);
    check(plugin_handle != nullptr, "test preloads legacy plugin fixture for counter inspection");
    if (plugin_handle == nullptr)
        return 1;
    auto count = reinterpret_cast<CountFn>(
        require_symbol(plugin_handle, "trtmc_test_legacy_plugin_create_calls"));
    auto reset =
        reinterpret_cast<ResetFn>(require_symbol(plugin_handle, "trtmc_test_legacy_plugin_reset"));
    reset();

    bool rejected_before_backend = false;
    try {
        (void)trtmc::load(dynamic_bundle.string(), options);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        rejected_before_backend =
            message.find("trtmc_model_plugin_query_abi_contract_v2") != std::string::npos &&
            message.find("before model-id/registration") != std::string::npos &&
            message.find("Backend") == std::string::npos;
    }
    check(rejected_before_backend, "new core rejects stale legacy plugin before backend dispatch");
    check(count() == 0, "dynamic rejection did not call stale plugin factory");

    void* backend_before = dlopen(backend_dso.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    check(backend_before == nullptr, "dynamic rejection occurs before backend DSO load");
    if (backend_before != nullptr)
        dlclose(backend_before);

    trtmc::LoadOptions legacy_options;
    legacy_options.model_plugin_search_paths = {plugin_dso.parent_path().string()};
    legacy_options.backend_search_paths = {backend_dso.parent_path().string()};
    bool static_succeeded = true;
    try {
        auto pipeline = trtmc::load(static_bundle.string(), legacy_options);
        check(pipeline == nullptr, "fixture legacy create returns its documented null sentinel");
    } catch (const std::exception& error) {
        std::cerr << "static compatibility path threw: " << error.what() << '\n';
        static_succeeded = false;
    }
    check(static_succeeded, "static bundle retains the legacy model-plugin path");
    check(count() == 1, "static bundle called the legacy plugin factory exactly once");

    void* backend_after = dlopen(backend_dso.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    check(backend_after != nullptr, "static legacy path loaded the test backend DSO");
    if (backend_after != nullptr)
        dlclose(backend_after);

    bool registered_legacy_rejected = false;
    try {
        (void)trtmc::load(dynamic_bundle.string(), options);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        registered_legacy_rejected =
            message.find("already registered") != std::string::npos &&
            message.find("without a verified") != std::string::npos &&
            message.find("before runtime-memory plugin dispatch") != std::string::npos;
    }
    check(registered_legacy_rejected,
          "a prior static legacy load cannot satisfy a later dynamic request");
    check(count() == 1, "dynamic retry did not call the legacy plugin factory");

    dlclose(plugin_handle);
    unsetenv("TRTMC_MODEL_PLUGIN_STRICT");

    if (failures == 0)
        std::cout << "legacy static compatibility and dynamic fail-closed ABI gate: PASS\n";
    return failures == 0 ? 0 : 1;
}
