/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for PipelineRegistry: register, lookup, unknown-strategy.
// Trace: ARCH-PIPELINE-REGISTRY, UD-REGISTRY-DISPATCH
// Intent: Verify registry-based pipeline plugin dispatch mechanics.
// Preconditions: No plugins registered at test start (fresh registry not possible
//   with singleton, so we test with known dummy strategies).
// Postconditions: Registry correctly maps strategies to plugins, returns nullptr
//   for unknown strategies, and lists registered strategies.

#include "trtmc/runtime/pipeline_registry.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

// Dummy plugin for testing
class DummyPlugin : public trtmc::IPipelinePlugin {
public:
    std::unique_ptr<trtmc::IPipeline> create(const trtmc::PipelineContext&) override
    {
        return nullptr; // Not testing actual pipeline creation
    }
};

// Test: register and lookup a plugin
static void test_register_and_lookup()
{
    static DummyPlugin plugin;
    auto& reg = trtmc::PipelineRegistry::instance();
    reg.register_plugin("__test_dummy_strategy__", &plugin);

    auto* found = reg.lookup("__test_dummy_strategy__");
    check(found == &plugin, "lookup returns registered plugin");
}

// Test: lookup unknown strategy returns nullptr
static void test_lookup_unknown_returns_nullptr()
{
    auto& reg = trtmc::PipelineRegistry::instance();
    auto* found = reg.lookup("__nonexistent_strategy_xyz__");
    check(found == nullptr, "lookup unknown strategy returns nullptr");
}

// Test: registered_strategies includes our test strategy
static void test_registered_strategies_includes_test()
{
    auto& reg = trtmc::PipelineRegistry::instance();
    auto strategies = reg.registered_strategies();
    bool found = std::find(strategies.begin(), strategies.end(),
                           "__test_dummy_strategy__") != strategies.end();
    check(found, "registered_strategies includes test strategy");
}

// Test: registering same strategy twice overwrites (last writer wins)
static void test_overwrite_registration()
{
    static DummyPlugin plugin_a;
    static DummyPlugin plugin_b;
    auto& reg = trtmc::PipelineRegistry::instance();
    reg.register_plugin("__test_overwrite__", &plugin_a);
    reg.register_plugin("__test_overwrite__", &plugin_b);
    auto* found = reg.lookup("__test_overwrite__");
    check(found == &plugin_b, "overwrite: last registration wins");
}

// Test: empty strategy string is rejected
static void test_empty_strategy_rejected()
{
    static DummyPlugin plugin;
    auto& reg = trtmc::PipelineRegistry::instance();
    bool threw = false;
    try {
        reg.register_plugin("", &plugin);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty strategy string throws invalid_argument");
}

// Test: null plugin is rejected
static void test_null_plugin_rejected()
{
    auto& reg = trtmc::PipelineRegistry::instance();
    bool threw = false;
    try {
        reg.register_plugin("__test_null__", nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "null plugin throws invalid_argument");
}

int main()
{
    test_register_and_lookup();
    test_lookup_unknown_returns_nullptr();
    test_registered_strategies_includes_test();
    test_overwrite_registration();
    test_empty_strategy_rejected();
    test_null_plugin_rejected();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    std::cerr << "All pipeline_registry tests passed" << std::endl;
    return 0;
}
