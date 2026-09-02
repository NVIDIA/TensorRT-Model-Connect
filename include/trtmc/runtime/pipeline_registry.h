/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Pipeline registry: singleton that maps runtime_strategy strings to
// IPipelinePlugin instances.

#include "trtmc/runtime/pipeline_plugin.h"

#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trtmc {

class PipelineRegistry {
  public:
    static PipelineRegistry& instance();

    // Register a plugin for one or more strategy strings. Built-in plugins
    // arrive through generated manifest registrar calls.
    void register_plugin(const std::string& strategy, IPipelinePlugin* plugin);

    // Look up the plugin for a given strategy string.
    // Returns nullptr if no plugin is registered for the strategy.
    IPipelinePlugin* lookup(const std::string& strategy) const;

    // Return all registered strategy strings (for diagnostics / testing).
    std::vector<std::string> registered_strategies() const;

  private:
    PipelineRegistry() = default;
    void begin_locked_registration(const std::string& strategy);
    void finish_locked_registration(const std::string& strategy);
    void abort_locked_registration(const std::string& strategy) noexcept;

    friend void load_model_plugin_for_strategy(const std::string& strategy,
                                               const std::vector<std::string>& search_paths);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, IPipelinePlugin*> registry_;
    bool locked_registration_open_{false};
    bool locked_registration_sealed_{false};
    std::thread::id locked_registration_thread_{};
    std::string locked_registration_strategy_;
    std::size_t locked_registration_count_{0};
};

// Legacy helper for ad hoc tests or local extensions that still rely on
// file-scope registration. Built-in plugins use manifest registration below.
struct PluginRegistrar {
    PluginRegistrar(const std::string& strategy, IPipelinePlugin* plugin) {
        PipelineRegistry::instance().register_plugin(strategy, plugin);
    }
};

// Legacy macro: declare a static plugin instance and register it for one
// strategy at process startup.
// Usage (at file scope in a plugin .cpp):
//   REGISTER_PIPELINE_PLUGIN("example_strategy", ExamplePlugin);
//
// For plugins handling multiple strategies, use the multi variant.
#define REGISTER_PIPELINE_PLUGIN(strategy, PluginClass)                                            \
    REGISTER_PIPELINE_PLUGIN_MULTI(PluginClass, strategy)

// Legacy multi-strategy variant: register same instance for multiple strategies.
// Usage:
//   REGISTER_PIPELINE_PLUGIN_MULTI(ExamplePlugin, "example_strategy", "alternate_strategy");
#define REGISTER_PIPELINE_PLUGIN_MULTI(PluginClass, ...)                                           \
    static PluginClass g_##PluginClass##_instance;                                                 \
    namespace {                                                                                    \
    struct PluginClass##_MultiReg {                                                                \
        PluginClass##_MultiReg() {                                                                 \
            for (const char* s : std::initializer_list<const char*>{__VA_ARGS__})                  \
                ::trtmc::PipelineRegistry::instance().register_plugin(                             \
                    s, &g_##PluginClass##_instance);                                               \
        }                                                                                          \
    };                                                                                             \
    static PluginClass##_MultiReg g_##PluginClass##_multi_reg;                                     \
    }

// Define the registration function consumed by the generated manifest source.
// The function name must match cmake/trtmc_pipeline_plugins.cmake.
#define REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(RegisterFunction, PluginClass, ...)                 \
    void RegisterFunction(::trtmc::PipelineRegistry& registry) {                                   \
        static PluginClass plugin_instance;                                                        \
        for (const char* s : std::initializer_list<const char*>{__VA_ARGS__})                      \
            registry.register_plugin(s, &plugin_instance);                                         \
    }

} // namespace trtmc
