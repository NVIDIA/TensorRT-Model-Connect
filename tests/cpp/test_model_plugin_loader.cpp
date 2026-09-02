/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for runtime model plugin lookup/loading.

#include "trtmc/runtime/pipeline_plugin_loader.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

static int failures = 0;

class DummyPlugin final : public trtmc::IPipelinePlugin {
  public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext&) override {
        return nullptr;
    }
};

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

static bool contains(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

#ifndef _WIN32
static void set_test_environment(const char* name, const char* value) {
    if (setenv(name, value, 1) != 0)
        throw std::runtime_error(std::string("Unable to set test environment variable: ") + name);
}

static void unset_test_environment(const char* name) {
    if (unsetenv(name) != 0)
        throw std::runtime_error(std::string("Unable to unset test environment variable: ") + name);
}
#endif

#ifdef _WIN32
struct WindowsEnvironmentValue {
    bool present{false};
    std::wstring value;
};

static WindowsEnvironmentValue windows_environment_value(const wchar_t* name) {
    std::vector<wchar_t> buffer(32768);
    SetLastError(ERROR_SUCCESS);
    const DWORD length =
        GetEnvironmentVariableW(name, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) {
        const DWORD error = GetLastError();
        if (error == ERROR_ENVVAR_NOT_FOUND)
            return {};
        if (error != ERROR_SUCCESS)
            throw std::runtime_error("Unable to read Windows test environment variable");
        return {true, {}};
    }
    if (length >= buffer.size())
        throw std::runtime_error("Windows test environment variable is too large");
    return {true, std::wstring(buffer.data(), length)};
}

class ScopedWindowsEnvironment final {
  public:
    ScopedWindowsEnvironment(const wchar_t* name, const wchar_t* value)
        : name_(name), previous_(windows_environment_value(name)) {
        if (!SetEnvironmentVariableW(name, value))
            throw std::runtime_error("Unable to set Windows test environment variable");
    }

    ~ScopedWindowsEnvironment() {
        SetEnvironmentVariableW(name_.c_str(),
                                previous_.present ? previous_.value.c_str() : nullptr);
    }

  private:
    std::wstring name_;
    WindowsEnvironmentValue previous_;
};

static std::wstring current_test_executable() {
    std::vector<wchar_t> buffer(32768);
    const DWORD length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size())
        throw std::runtime_error("Unable to locate the Windows test executable");
    return std::wstring(buffer.data(), length);
}

static DWORD run_strict_loading_probe_child() {
    ScopedWindowsEnvironment strict(L"TRTMC_MODEL_PLUGIN_STRICT", L"1");
    ScopedWindowsEnvironment no_directory(L"TRTMC_MODEL_PLUGIN_DIR", nullptr);

    const std::wstring executable = current_test_executable();
    std::wstring command_line = L"\"" + executable + L"\" --strict-loading-probe";
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(), mutable_command.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process)) {
        throw std::runtime_error("Unable to launch the Windows strict-loading probe");
    }

    const DWORD wait = WaitForSingleObject(process.hProcess, 30000);
    DWORD exit_code = std::numeric_limits<DWORD>::max();
    if (wait == WAIT_TIMEOUT) {
        TerminateProcess(process.hProcess, exit_code);
    } else if (wait != WAIT_OBJECT_0 || !GetExitCodeProcess(process.hProcess, &exit_code)) {
        exit_code = std::numeric_limits<DWORD>::max();
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return exit_code;
}
#endif

static const trtmc::ModelPluginInfo* first_index_entry() {
    for (const auto& entry : trtmc::runtime_model_plugin_index()) {
        if (entry.model_id != nullptr && entry.runtime_strategy != nullptr &&
            entry.library_name != nullptr) {
            return &entry;
        }
    }
    return nullptr;
}

static std::vector<std::string> strategies_for_model(const std::string& model_id) {
    std::vector<std::string> strategies;
    for (const auto& entry : trtmc::runtime_model_plugin_index()) {
        if (entry.model_id != nullptr && model_id == entry.model_id &&
            entry.runtime_strategy != nullptr) {
            strategies.emplace_back(entry.runtime_strategy);
        }
    }
    return strategies;
}

static void test_index_maps_strategy_to_model() {
    const auto* sample = first_index_entry();
    check(sample != nullptr, "plugin index has at least one entry");
    if (sample == nullptr)
        return;

    auto model = trtmc::model_plugin_id_for_strategy(sample->runtime_strategy);
    check(model.has_value(), "indexed strategy has model plugin");
    check(model && *model == sample->model_id, "indexed strategy maps to declared model");
    check(trtmc::model_plugin_library_name(sample->model_id) == sample->library_name,
          "indexed model library name");
}

static void test_registry_does_not_eager_register_models() {
    const auto* sample = first_index_entry();
    check(sample != nullptr, "plugin index has entry before eager-load check");
    if (sample == nullptr)
        return;

    auto* plugin = trtmc::PipelineRegistry::instance().lookup(sample->runtime_strategy);
    check(plugin == nullptr, "model plugin not registered before explicit load");
}

static void test_unknown_strategy_reports_clean_error() {
    bool threw = false;
    try {
        trtmc::load_model_plugin_for_strategy("__missing_strategy__");
    } catch (const std::runtime_error& e) {
        threw = true;
        check(std::string(e.what()).find("No plugin registered for runtime_strategy") !=
                  std::string::npos,
              "unknown strategy error uses public registry wording");
    }
    check(threw, "unknown strategy throws");
}

#if !defined(TRTMC_LOCKED_H3_RUNTIME)
static void test_strict_loading_requires_an_explicit_directory() {
    const auto* sample = first_index_entry();
    check(sample != nullptr, "plugin index has entry before strict-load check");
    if (sample == nullptr)
        return;

#ifdef _WIN32
    // /MT gives the test executable and trtmc_core.dll independent CRT
    // environment caches. Launch a child after changing the process
    // environment so every CRT observes the real inherited startup state.
    check(run_strict_loading_probe_child() == 0,
          "strict loading without a directory throws in a fresh process");
#else
    const char* previous_strict = std::getenv("TRTMC_MODEL_PLUGIN_STRICT");
    const char* previous_dir = std::getenv("TRTMC_MODEL_PLUGIN_DIR");
    const std::string saved_strict = previous_strict ? previous_strict : "";
    const std::string saved_dir = previous_dir ? previous_dir : "";
    const bool had_strict = previous_strict != nullptr;
    const bool had_dir = previous_dir != nullptr;

    set_test_environment("TRTMC_MODEL_PLUGIN_STRICT", "1");
    unset_test_environment("TRTMC_MODEL_PLUGIN_DIR");
    bool threw = false;
    try {
        trtmc::load_model_plugin_for_strategy(sample->runtime_strategy);
    } catch (const std::runtime_error& e) {
        threw = true;
        check(std::string(e.what()).find("requires an explicit model plugin search path") !=
                  std::string::npos,
              "strict loading reports missing explicit directory");
    }
    check(threw, "strict loading without a directory throws");

    if (had_strict)
        set_test_environment("TRTMC_MODEL_PLUGIN_STRICT", saved_strict.c_str());
    else
        unset_test_environment("TRTMC_MODEL_PLUGIN_STRICT");
    if (had_dir)
        set_test_environment("TRTMC_MODEL_PLUGIN_DIR", saved_dir.c_str());
    else
        unset_test_environment("TRTMC_MODEL_PLUGIN_DIR");
#endif
}
#else
static DummyPlugin g_untrusted_h3_plugin;

static void test_locked_registry_rejects_untrusted_first_writer() {
    const auto* sample = first_index_entry();
    check(sample != nullptr, "plugin index has entry before locked registry attack check");
    if (sample == nullptr)
        return;

    auto& registry = trtmc::PipelineRegistry::instance();
    bool rejected = false;
    try {
        registry.register_plugin(sample->runtime_strategy, &g_untrusted_h3_plugin);
    } catch (const std::runtime_error& error) {
        rejected =
            std::string(error.what()).find("outside its trusted loader") != std::string::npos;
    }
    check(rejected, "locked registry rejects an untrusted H3 first writer");
    check(registry.lookup(sample->runtime_strategy) == nullptr,
          "untrusted H3 first writer cannot poison the registry");
}

static void test_locked_loading_uses_only_the_packaged_path() {
    const auto* sample = first_index_entry();
    check(sample != nullptr, "plugin index has entry before locked-load check");
    if (sample == nullptr)
        return;

#ifdef _WIN32
    ScopedWindowsEnvironment injected_directory(L"TRTMC_MODEL_PLUGIN_DIR", L"C:\\untrusted");
    ScopedWindowsEnvironment injected_strict(L"TRTMC_MODEL_PLUGIN_STRICT", L"1");
#endif
    bool loaded = true;
    try {
        trtmc::load_model_plugin_for_strategy(sample->runtime_strategy);
    } catch (const std::runtime_error& error) {
        loaded = false;
        std::cerr << "FAIL: locked packaged plugin load: " << error.what() << '\n';
        ++failures;
    }
    check(loaded, "locked loader ignores plugin environment injection");

    bool override_rejected = false;
    try {
        trtmc::load_model_plugin_for_strategy(sample->runtime_strategy, {"C:\\untrusted"});
    } catch (const std::runtime_error& error) {
        override_rejected = true;
        check(std::string(error.what()).find("rejects model plugin search path overrides") !=
                  std::string::npos,
              "locked loader explains rejected plugin override");
    }
    check(override_rejected, "locked loader rejects explicit plugin path after loading");
}

static void test_locked_registry_rejects_post_load_replacement() {
    const auto* sample = first_index_entry();
    check(sample != nullptr, "plugin index has entry before replacement attack check");
    if (sample == nullptr)
        return;

    auto& registry = trtmc::PipelineRegistry::instance();
    auto* trusted = registry.lookup(sample->runtime_strategy);
    check(trusted != nullptr && trusted != &g_untrusted_h3_plugin,
          "trusted packaged H3 plugin owns the sealed strategy");
    bool rejected = false;
    try {
        registry.register_plugin(sample->runtime_strategy, &g_untrusted_h3_plugin);
    } catch (const std::runtime_error& error) {
        rejected =
            std::string(error.what()).find("outside its trusted loader") != std::string::npos;
    }
    check(rejected, "locked registry rejects post-load H3 replacement");
    check(registry.lookup(sample->runtime_strategy) == trusted,
          "post-load replacement cannot change the sealed H3 plugin");
}
#endif

static void test_load_index_owner_registers_only_that_model() {
    const auto* sample = first_index_entry();
    check(sample != nullptr, "plugin index has entry before load check");
    if (sample == nullptr)
        return;

    const std::string owner = sample->model_id;
    trtmc::load_model_plugin_for_strategy(sample->runtime_strategy);
    auto strategies = trtmc::PipelineRegistry::instance().registered_strategies();
    for (const auto& expected : strategies_for_model(owner)) {
        check(contains(strategies, expected), "owning model strategy registered");
    }
    bool saw_unrelated_model = false;
    for (const auto& strategy : strategies) {
        auto model = trtmc::model_plugin_id_for_strategy(strategy);
        if (model && *model != owner) {
            saw_unrelated_model = true;
        }
    }
    check(!saw_unrelated_model, "unrelated model plugin not registered");
}

int main(int argc, char** argv) {
#ifdef _WIN32
    if (argc == 2 && std::string(argv[1]) == "--strict-loading-probe") {
        const auto* sample = first_index_entry();
        if (sample == nullptr)
            return 2;
        try {
            trtmc::load_model_plugin_for_strategy(sample->runtime_strategy);
        } catch (const std::runtime_error& error) {
            return std::string(error.what())
                               .find("requires an explicit model plugin search path") !=
                           std::string::npos
                       ? 0
                       : 3;
        }
        return 4;
    }
#else
    (void)argc;
    (void)argv;
#endif
    test_index_maps_strategy_to_model();
    test_registry_does_not_eager_register_models();
    test_unknown_strategy_reports_clean_error();
#if defined(TRTMC_LOCKED_H3_RUNTIME)
    test_locked_registry_rejects_untrusted_first_writer();
    test_locked_loading_uses_only_the_packaged_path();
    test_locked_registry_rejects_post_load_replacement();
#else
    test_strict_loading_requires_an_explicit_directory();
#endif
    test_load_index_owner_registers_only_that_model();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    std::cerr << "All model_plugin_loader tests passed" << std::endl;
    return 0;
}
