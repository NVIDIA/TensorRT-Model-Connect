/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/config/cli_support.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
void check_invalid(Callable&& callable, const char* label) {
    try {
        callable();
        check(false, label);
    } catch (const std::invalid_argument&) {
    }
}

} // namespace

int main() {
    using trtmc::config::ConfigBundle;
    using trtmc::config::Layer;
    using trtmc::config::SchemaRegistry;

    auto& schemas = SchemaRegistry::instance();
    check(schemas.lookup("minimax_h3") == nullptr, "minimax_h3 schema absent from core");

    trtmc::load_model_plugin_for_strategy("diffusion_minimax_h3");
    const auto* schema = schemas.lookup("minimax_h3");
    check(schema != nullptr, "MiniMax-H3 plugin registers model-owned schema");
    check(schema != nullptr && schema->fields.size() == 1, "MiniMax-H3 schema field count");

    const ConfigBundle defaults = ConfigBundle::build({}, schemas);
    check(defaults.source_of("minimax_h3", "first_block_cache_threshold") == Layer::SchemaDefault,
          "cache threshold default retains SchemaDefault provenance");
    check(std::abs(defaults.get<double>("minimax_h3", "first_block_cache_threshold") - 0.025) <
              1.0e-12,
          "cache threshold schema default");

    const ConfigBundle overridden = trtmc::config::resolve_cli_config(
        "", {"minimax_h3.first_block_cache_threshold=0.05"}, {}, schemas);
    check(overridden.source_of("minimax_h3", "first_block_cache_threshold") ==
              Layer::SessionRequest,
          "qualified cache threshold records session provenance");
    check(std::abs(overridden.get<double>("minimax_h3", "first_block_cache_threshold") - 0.05) <
              1.0e-12,
          "qualified cache threshold reaches resolved config");

    check_invalid(
        [&] {
            (void)trtmc::config::resolve_cli_config("", {"first_block_cache_threshold=0.05"}, {},
                                                    schemas);
        },
        "unqualified cache threshold fails closed");
    for (const std::string& value : {"0", "-0.01", "nan", "inf"}) {
        check_invalid(
            [&] {
                (void)trtmc::config::resolve_cli_config(
                    "", {"minimax_h3.first_block_cache_threshold=" + value}, {}, schemas);
            },
            "non-positive or non-finite cache threshold fails closed");
    }

    if (failures > 0) {
        std::cerr << failures << " MiniMax-H3 config schema test(s) failed\n";
        return 1;
    }
    return 0;
}
