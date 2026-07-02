/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CFG-TRIATTN-CPP-01
// Architecture:   ARCH-CFG-001
// Unit Design:    UD-CFG-REG-01
// Intent:         Verify the triattention schema manifest registration
//                 survives the static-library link.
// Preconditions:  libtrtmc_core linked; no prior SchemaRegistry mutation.
// Postconditions: SchemaRegistry::instance().lookup("triattention") returns
//                 a schema whose field set matches the declared fields.
// =============================================================================

#include "trtmc/config/schema_registry.h"
#include "trtmc/config/schemas/triattention.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

using trtmc::config::Layer;
using trtmc::config::Schema;
using trtmc::config::SchemaRegistry;

void test_manifest_registered_triattention() {
    // The generated schema registrar references triattention.cpp directly,
    // so it is linked and registered before the lookup below.
    const Schema* schema = SchemaRegistry::instance().lookup("triattention");
    check(schema != nullptr, "manifest: triattention is registered");
    if (schema == nullptr)
        return;

    // Spot-check a handful of expected fields.
    const std::set<std::string> expected = {
        "enabled",
        "kv_budget",
        "divide_length",
        "recent_window",
        "score_aggregation",
        "per_layer_aggregation",
        "count_prompt_tokens",
        "protect_prefill",
        "disable_mlr",
        "disable_trig",
        "offset_max_length",
        "stats_section",
        "debug",
        "profile",
        "runtime_bucket_rows",
        "disable_gpu_selection",
        "disable_gpu_compaction",
        "disable_gpu_state",
        "zero_tail",
        "dump_keep_path",
        "dump_compaction_index",
        "abort_after_dump",
        "dump_score_cache",
        "dump_score_values",
    };
    std::set<std::string> actual;
    for (const auto& f : schema->fields)
        actual.insert(f.name);

    for (const auto& name : expected) {
        if (actual.count(name) == 0) {
            std::cerr << "FAIL: missing expected field: " << name << '\n';
            ++g_failures;
        }
    }
    for (const auto& name : actual) {
        if (expected.count(name) == 0) {
            std::cerr << "FAIL: unexpected field in schema: " << name << '\n';
            ++g_failures;
        }
    }
    check(actual.size() == expected.size(), "triattention: field count matches");
}

void test_make_triattention_schema_standalone() {
    // Construct without using the singleton — helps diagnose whether a
    // failure in the other test is a registration problem or a declaration
    // problem.
    Schema schema = trtmc::config::schemas::make_triattention_schema();
    check(schema.namespace_name == "triattention", "make_schema: namespace");
    check(schema.fields.size() >= 20, "make_schema: at least 20 fields");
    // kv_budget field must be int32 with default 4096 and allow Bundle+Session.
    const auto it = std::find_if(schema.fields.begin(), schema.fields.end(),
                                 [](const auto& f) { return f.name == "kv_budget"; });
    check(it != schema.fields.end(), "make_schema: kv_budget present");
    if (it != schema.fields.end()) {
        check(it->type == "int32", "make_schema: kv_budget type");
        check(it->allowed_layers.count(Layer::SessionRequest) != 0,
              "make_schema: kv_budget allows session");
        check(it->allowed_layers.count(Layer::BundleDefault) != 0,
              "make_schema: kv_budget allows bundle default");
        check(std::any_cast<std::int32_t>(it->default_value) == 4096,
              "make_schema: kv_budget default");
    }
}

} // namespace

int main() {
    test_manifest_registered_triattention();
    test_make_triattention_schema_standalone();
    if (g_failures != 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
