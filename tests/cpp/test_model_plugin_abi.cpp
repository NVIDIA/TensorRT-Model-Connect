/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/registry/model_plugin_abi.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef TRTMC_TEST_STALE_MODEL_PLUGIN_DSO
#error "TRTMC_TEST_STALE_MODEL_PLUGIN_DSO must name the stale model fixture"
#endif
#ifndef TRTMC_TEST_PAIRED_MODEL_PLUGIN_DSO
#error "TRTMC_TEST_PAIRED_MODEL_PLUGIN_DSO must name the paired model fixture"
#endif
#ifndef TRTMC_TEST_STALE_MODEL_FINGERPRINT_DSO
#error "TRTMC_TEST_STALE_MODEL_FINGERPRINT_DSO must name the bad-fingerprint fixture"
#endif

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

using CountFn = std::uint32_t (*)();

} // namespace

int main() {
    setenv("TRTMC_MODEL_PLUGIN_STRICT", "1", 1);
    unsetenv("TRTMC_MODEL_PLUGIN_DIR");

    void* stale = dlopen(TRTMC_TEST_STALE_MODEL_PLUGIN_DSO, RTLD_NOW | RTLD_LOCAL);
    if (stale == nullptr) {
        std::cerr << "FAIL: could not open stale model fixture: " << dlerror() << '\n';
        return 1;
    }
    auto stale_id =
        reinterpret_cast<CountFn>(require_symbol(stale, "trtmc_test_stale_model_id_calls"));
    auto stale_registration = reinterpret_cast<CountFn>(
        require_symbol(stale, "trtmc_test_stale_model_registration_calls"));
    auto stale_factory =
        reinterpret_cast<CountFn>(require_symbol(stale, "trtmc_test_stale_model_factory_calls"));

    bool stale_rejected = false;
    try {
        const std::filesystem::path path(TRTMC_TEST_STALE_MODEL_PLUGIN_DSO);
        trtmc::load_model_plugin_for_strategy("qwen_decoder_kv_cache",
                                              {path.parent_path().string()});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        stale_rejected =
            message.find(trtmc::kModelPluginDsoAbiQuerySymbolV2) != std::string::npos &&
            message.find("before model-id/registration") != std::string::npos;
    }
    check(stale_rejected, "stale model plugin is rejected by the pre-registration ABI gate");
    check(stale_id() == 0, "stale model-id entrypoint was not called");
    check(stale_registration() == 0, "stale model registration was not called");
    check(stale_factory() == 0, "stale model factory was not called");
    dlclose(stale);

    void* stale_fingerprint = dlopen(TRTMC_TEST_STALE_MODEL_FINGERPRINT_DSO, RTLD_NOW | RTLD_LOCAL);
    if (stale_fingerprint == nullptr) {
        std::cerr << "FAIL: could not open bad-fingerprint model fixture: " << dlerror() << '\n';
        return 1;
    }
    auto stale_fingerprint_query = reinterpret_cast<CountFn>(
        require_symbol(stale_fingerprint, "trtmc_test_paired_model_query_calls"));
    auto stale_fingerprint_id = reinterpret_cast<CountFn>(
        require_symbol(stale_fingerprint, "trtmc_test_paired_model_id_calls"));
    auto stale_fingerprint_registration = reinterpret_cast<CountFn>(
        require_symbol(stale_fingerprint, "trtmc_test_paired_model_registration_calls"));
    auto stale_fingerprint_factory = reinterpret_cast<CountFn>(
        require_symbol(stale_fingerprint, "trtmc_test_paired_model_factory_calls"));

    bool fingerprint_rejected = false;
    try {
        const std::filesystem::path path(TRTMC_TEST_STALE_MODEL_FINGERPRINT_DSO);
        trtmc::load_model_plugin_for_strategy("qwen_decoder_kv_cache",
                                              {path.parent_path().string()});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        fingerprint_rejected = message.find("interface_fingerprint") != std::string::npos &&
                               message.find("before model-id/registration") != std::string::npos;
    }
    check(fingerprint_rejected, "bad model-plugin fingerprint is rejected");
    check(stale_fingerprint_query() == 1, "bad fingerprint query ran exactly once");
    check(stale_fingerprint_id() == 0, "bad fingerprint model-id was not called");
    check(stale_fingerprint_registration() == 0,
          "bad fingerprint model registration was not called");
    check(stale_fingerprint_factory() == 0, "bad fingerprint model factory was not called");
    dlclose(stale_fingerprint);

    void* paired = dlopen(TRTMC_TEST_PAIRED_MODEL_PLUGIN_DSO, RTLD_NOW | RTLD_LOCAL);
    if (paired == nullptr) {
        std::cerr << "FAIL: could not open paired model fixture: " << dlerror() << '\n';
        return 1;
    }
    auto paired_query =
        reinterpret_cast<CountFn>(require_symbol(paired, "trtmc_test_paired_model_query_calls"));
    auto paired_id =
        reinterpret_cast<CountFn>(require_symbol(paired, "trtmc_test_paired_model_id_calls"));
    auto paired_registration = reinterpret_cast<CountFn>(
        require_symbol(paired, "trtmc_test_paired_model_registration_calls"));
    auto paired_factory =
        reinterpret_cast<CountFn>(require_symbol(paired, "trtmc_test_paired_model_factory_calls"));

    const std::filesystem::path paired_path(TRTMC_TEST_PAIRED_MODEL_PLUGIN_DSO);
    trtmc::load_model_plugin_for_strategy("qwen_decoder_kv_cache",
                                          {paired_path.parent_path().string()});
    check(paired_query() == 1, "paired model ABI query ran exactly once");
    check(paired_id() == 1, "paired model id ran after the ABI query");
    check(paired_registration() == 1, "paired model registered exactly once");
    check(paired_factory() == 0, "paired model factory is not called during registration");

    auto* plugin = trtmc::PipelineRegistry::instance().lookup("qwen_decoder_kv_cache");
    check(plugin != nullptr, "paired model plugin is available after registration");
    if (plugin != nullptr) {
        trtmc::BundleFile bundle;
        trtmc::BaseConfig config;
        const std::string empty;
        const trtmc::PipelineContext context{bundle,  config, empty, empty, empty,
                                             nullptr, empty,  false, 0,     nullptr};
        auto pipeline = plugin->create(context);
        check(pipeline == nullptr, "paired fixture returns its documented null sentinel");
    }
    check(paired_factory() == 1, "paired model factory remains callable after the ABI gate");

    dlclose(paired);
    unsetenv("TRTMC_MODEL_PLUGIN_STRICT");

    std::cerr << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << '\n';
    return failures;
}
