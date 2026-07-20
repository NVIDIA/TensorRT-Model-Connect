/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_plugin_loader.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void restore_environment(const char* name, bool existed, const std::string& value) {
    if (existed)
        setenv(name, value.c_str(), 1);
    else
        unsetenv(name);
}

void test_previous_model_plugin_abi_is_rejected() {
    namespace fs = std::filesystem;
    constexpr const char* kModelId = "wan2_2_ti2v";
    constexpr const char* kStrategy = "diffusion_wan2_2_ti2v";

    std::error_code error;
    const auto executable = fs::read_symlink("/proc/self/exe", error);
    const auto fixture = executable.parent_path() / "test_model_plugins" /
                         "libtrtmc_test_model_plugin_api_abi_v1.so";
    const auto library_name = trtmc::model_plugin_library_name(kModelId);
    check(!error && !library_name.empty() && fs::is_regular_file(fixture),
          "Wan2.2 model plugin and previous-ABI fixture exist");

    const fs::path test_root =
        fs::temp_directory_path() / ("trtmc-wan22-model-plugin-v1-" + std::to_string(getpid()));
    std::error_code cleanup_error;
    fs::remove_all(test_root, cleanup_error);
    const fs::path plugin_dir = test_root / kModelId;
    fs::create_directories(plugin_dir);
    fs::create_symlink(fixture, plugin_dir / library_name);

    const char* previous_strict = std::getenv("TRTMC_MODEL_PLUGIN_STRICT");
    const char* previous_dir = std::getenv("TRTMC_MODEL_PLUGIN_DIR");
    const std::string saved_strict = previous_strict != nullptr ? previous_strict : "";
    const std::string saved_dir = previous_dir != nullptr ? previous_dir : "";
    const bool had_strict = previous_strict != nullptr;
    const bool had_dir = previous_dir != nullptr;

    setenv("TRTMC_MODEL_PLUGIN_STRICT", "1", 1);
    unsetenv("TRTMC_MODEL_PLUGIN_DIR");
    bool threw = false;
    try {
        trtmc::load_model_plugin_for_strategy(kStrategy, {test_root.string()});
    } catch (const std::runtime_error& exception) {
        threw = true;
        const std::string message = exception.what();
        check(message.find("model plugin API ABI 1") != std::string::npos,
              "old model plugin ABI is reported");
        check(
            message.find("runtime ABI " + std::to_string(trtmc::kTrtmcModelPluginApiAbiVersion)) !=
                std::string::npos,
            "required model plugin ABI is reported");
    }
    check(threw, "previous model plugin ABI is rejected before registration");

    restore_environment("TRTMC_MODEL_PLUGIN_STRICT", had_strict, saved_strict);
    restore_environment("TRTMC_MODEL_PLUGIN_DIR", had_dir, saved_dir);
    fs::remove_all(test_root, cleanup_error);
}

} // namespace

int main() {
    test_previous_model_plugin_abi_is_rejected();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "Wan2.2 model plugin ABI test passed\n";
    return 0;
}
