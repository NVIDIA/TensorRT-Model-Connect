/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "frozen_github_main_abi.h"

#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

#ifndef TRTMC_TEST_CURRENT_PLUGIN_DSO
#error "TRTMC_TEST_CURRENT_PLUGIN_DSO must name the independently built current plugin"
#endif

// The old plugin header intentionally forward-declared this internal type.
// The legacy harness needs only an address because neither side dereferences it.
namespace trtmc {
struct BundleFile {};
} // namespace trtmc

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void* require_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr)
        throw std::runtime_error(std::string("missing fixture symbol ") + name);
    return symbol;
}

using PluginFn = trtmc::IPipelinePlugin* (*)();
using CountFn = std::uint32_t (*)();
using ResetFn = void (*)();

trtmc::PipelineContext make_context(const trtmc::BundleFile& bundle,
                                    const trtmc::BaseConfig& config, const std::string& config_json,
                                    const std::string& bundle_path, const std::string& empty) {
    return {bundle, config, config_json, empty, bundle_path, nullptr, empty, false, 0, nullptr};
}

class TempDirectory {
  public:
    TempDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("trtmc-legacy-core-abi-" + std::to_string(static_cast<long long>(getpid())));
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

void write_bundle_header(const std::filesystem::path& path, bool dynamic) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << "{\"model_id\":\"qwen\"";
    if (dynamic)
        output << ",\"runtime_memory\":{\"contract_version\":1}";
    output << '}';
}

} // namespace

int main() {
    TempDirectory temp;
    const auto static_bundle_path = temp.path() / "static.trtfb";
    const auto dynamic_bundle_path = temp.path() / "dynamic.trtfb";
    write_bundle_header(static_bundle_path, false);
    write_bundle_header(dynamic_bundle_path, true);

    void* handle = dlopen(TRTMC_TEST_CURRENT_PLUGIN_DSO, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << "FAIL: dlopen current plugin: " << dlerror() << '\n';
        return 1;
    }

    auto get_plugin =
        reinterpret_cast<PluginFn>(require_symbol(handle, "trtmc_test_current_plugin_base_v1"));
    auto legacy_count = reinterpret_cast<CountFn>(
        require_symbol(handle, "trtmc_test_current_plugin_legacy_create_calls"));
    auto runtime_count = reinterpret_cast<CountFn>(
        require_symbol(handle, "trtmc_test_current_plugin_runtime_create_calls"));
    auto reset =
        reinterpret_cast<ResetFn>(require_symbol(handle, "trtmc_test_current_plugin_reset"));

    trtmc::IPipelinePlugin* plugin = get_plugin();
    check(plugin != nullptr, "legacy harness obtained the stable base plugin interface");
    reset();

    trtmc::BundleFile bundle;
    trtmc::BaseConfig config;
    const std::string empty;
    const std::string static_config = R"({"runtime_strategy":"qwen_decoder_kv_cache"})";
    const std::string static_bundle_path_text = static_bundle_path.string();
    const auto static_context =
        make_context(bundle, config, static_config, static_bundle_path_text, empty);

    bool static_succeeded = true;
    try {
        auto pipeline = plugin->create(static_context);
        check(pipeline == nullptr, "current fixture returns its documented null sentinel");
    } catch (const std::exception& error) {
        std::cerr << "static legacy-core path threw: " << error.what() << '\n';
        static_succeeded = false;
    }
    check(static_succeeded, "old core can call current plugin's stable create slot");
    check(legacy_count() == 1 && runtime_count() == 0,
          "static old-core path used only the stable base create slot");

    const std::string dynamic_config = R"({"runtime_strategy":"qwen_decoder_kv_cache"})";
    const std::string dynamic_bundle_path_text = dynamic_bundle_path.string();
    const auto dynamic_context =
        make_context(bundle, config, dynamic_config, dynamic_bundle_path_text, empty);
    bool dynamic_rejected = false;
    try {
        (void)plugin->create(dynamic_context);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        dynamic_rejected = message.find("legacy core fail closed") != std::string::npos;
    }
    check(dynamic_rejected,
          "current plugin base create explicitly rejects runtime_memory from a legacy core");
    check(legacy_count() == 1 && runtime_count() == 0,
          "legacy core cannot accidentally enter the new runtime-memory slot");

    dlclose(handle);
    if (failures == 0)
        std::cout << "current plugin/legacy core DSO compatibility: PASS\n";
    return failures == 0 ? 0 : 1;
}
