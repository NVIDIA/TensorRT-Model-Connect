/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/hot_engine_policy.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"

#include <cmath>
#include <cstdint>
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

    bool all_fast_h3_transitions_retained = true;
    int retained_transition_count = 0;
    for (int index = 0; index < 100; ++index) {
        std::string name = "denoiser_transition_";
        name.push_back(static_cast<char>('0' + index / 10));
        name.push_back(static_cast<char>('0' + index % 10));
        name += "_plan";
        const bool retained = trtmc::minimax_h3::should_retain_hot_engine(name, true);
        retained_transition_count += retained ? 1 : 0;
        if (index < 49)
            all_fast_h3_transitions_retained &= retained;
    }
    check(all_fast_h3_transitions_retained, "all 49 FastH3 transition engines are retained");
    check(retained_transition_count == 49,
          "retained transition policy is limited to the authenticated 00..48 range");
    for (std::string_view name : {"denoiser_entry_plan", "denoiser_finish_plan",
                                  "vae_tile_decoder_plan", "audio_vae_decoder_plan"}) {
        check(trtmc::minimax_h3::should_retain_hot_engine(name, true),
              "FastH3 staged hot engine is retained");
        check(!trtmc::minimax_h3::should_retain_hot_engine(name, false),
              "FastH3 staged engine retention remains opt-in");
    }
    for (std::string_view name : {"denoiser_transition_49_plan", "denoiser_transition_0_plan",
                                  "text_encoder_plan", "adaln_precompute_plan"}) {
        check(!trtmc::minimax_h3::should_retain_hot_engine(name, true),
              "non-FastH3 engine is excluded from the retained hot set");
    }

    auto& schemas = SchemaRegistry::instance();
    check(schemas.lookup("minimax_h3") == nullptr, "minimax_h3 schema absent from core");

    trtmc::load_model_plugin_for_strategy("diffusion_minimax_h3");
    const auto* schema = schemas.lookup("minimax_h3");
    check(schema != nullptr, "MiniMax-H3 plugin registers model-owned schema");
    check(schema != nullptr && schema->fields.size() == 3, "MiniMax-H3 schema field count");

    const ConfigBundle defaults = ConfigBundle::build({}, schemas);
    check(defaults.source_of("minimax_h3", "first_block_cache_threshold") == Layer::SchemaDefault,
          "cache threshold default retains SchemaDefault provenance");
    check(std::abs(defaults.get<double>("minimax_h3", "first_block_cache_threshold") - 0.025) <
              1.0e-12,
          "cache threshold schema default");
    check(!defaults.get<bool>("minimax_h3", "retain_engines"), "retained engines are opt-in");
    check(defaults.get<std::int64_t>("minimax_h3", "retained_tail_weight_budget_gib") == 24,
          "retained tail budget schema default");

    const ConfigBundle overridden = trtmc::config::resolve_cli_config(
        "",
        {"minimax_h3.first_block_cache_threshold=0.05", "minimax_h3.retain_engines=true",
         "minimax_h3.retained_tail_weight_budget_gib=30"},
        {}, schemas);
    check(overridden.source_of("minimax_h3", "first_block_cache_threshold") ==
              Layer::SessionRequest,
          "qualified cache threshold records session provenance");
    check(std::abs(overridden.get<double>("minimax_h3", "first_block_cache_threshold") - 0.05) <
              1.0e-12,
          "qualified cache threshold reaches resolved config");
    check(overridden.get<bool>("minimax_h3", "retain_engines"),
          "qualified retained-engine flag reaches resolved config");
    check(overridden.get<std::int64_t>("minimax_h3", "retained_tail_weight_budget_gib") == 30,
          "qualified retained tail budget reaches resolved config");

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
    for (const std::string& value : {"0", "-1", "8589934592"}) {
        check_invalid(
            [&] {
                (void)trtmc::config::resolve_cli_config(
                    "", {"minimax_h3.retained_tail_weight_budget_gib=" + value}, {}, schemas);
            },
            "invalid retained tail budget fails closed");
    }

    if (failures > 0) {
        std::cerr << failures << " MiniMax-H3 config schema test(s) failed\n";
        return 1;
    }
    return 0;
}
