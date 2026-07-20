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
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

static bool contains(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

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

static void test_strict_loading_requires_an_explicit_directory() {
    const auto* sample = first_index_entry();
    check(sample != nullptr, "plugin index has entry before strict-load check");
    if (sample == nullptr)
        return;

    const char* previous_strict = std::getenv("TRTMC_MODEL_PLUGIN_STRICT");
    const char* previous_dir = std::getenv("TRTMC_MODEL_PLUGIN_DIR");
    const std::string saved_strict = previous_strict ? previous_strict : "";
    const std::string saved_dir = previous_dir ? previous_dir : "";
    const bool had_strict = previous_strict != nullptr;
    const bool had_dir = previous_dir != nullptr;

    setenv("TRTMC_MODEL_PLUGIN_STRICT", "1", 1);
    unsetenv("TRTMC_MODEL_PLUGIN_DIR");
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
        setenv("TRTMC_MODEL_PLUGIN_STRICT", saved_strict.c_str(), 1);
    else
        unsetenv("TRTMC_MODEL_PLUGIN_STRICT");
    if (had_dir)
        setenv("TRTMC_MODEL_PLUGIN_DIR", saved_dir.c_str(), 1);
    else
        unsetenv("TRTMC_MODEL_PLUGIN_DIR");
}

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

int main() {
    test_index_maps_strategy_to_model();
    test_registry_does_not_eager_register_models();
    test_unknown_strategy_reports_clean_error();
    test_strict_loading_requires_an_explicit_directory();
    test_load_index_owner_registers_only_that_model();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    std::cerr << "All model_plugin_loader tests passed" << std::endl;
    return 0;
}
