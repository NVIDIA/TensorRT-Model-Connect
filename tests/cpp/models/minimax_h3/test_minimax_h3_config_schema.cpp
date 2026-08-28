/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

constexpr std::array<std::pair<const char*, int32_t>, 17> kRef2vaProfile = {{
    {"min_text_rows", 1},
    {"opt_text_rows", 8192},
    {"max_text_rows", 262144},
    {"ref2va_min_condition_video_rows", 0},
    {"ref2va_opt_condition_video_rows", 4096},
    {"ref2va_min_condition_audio_rows", 0},
    {"ref2va_opt_condition_audio_rows", 0},
    {"ref2va_max_condition_video_rows", 258120},
    {"ref2va_max_condition_audio_rows", 2408},
    {"ref2va_max_images", 9},
    {"ref2va_max_videos", 3},
    {"ref2va_max_audios", 3},
    {"ref2va_max_references", 12},
    {"ref2va_reference_min_seconds", 2},
    {"ref2va_reference_max_seconds", 15},
    {"ref2va_vae_tile_size", 256},
    {"ref2va_vae_tile_min_overlap", 64},
}};

std::string ref2va_profile_json(const std::string& omitted = {}) {
    std::string result = R"({"workflow":"ref2va")";
    for (const auto& [name, value] : kRef2vaProfile) {
        if (name != omitted)
            result += ",\"" + std::string(name) + "\":" + std::to_string(value);
    }
    result += "}";
    return result;
}

std::string create_error(const std::string& config_json) {
    auto* plugin = trtmc::PipelineRegistry::instance().lookup("diffusion_minimax_h3");
    if (plugin == nullptr)
        return "plugin is missing";
    const trtmc::BundleFile bundle;
    const trtmc::BaseConfig config;
    const std::string empty;
    const trtmc::PipelineContext context{
        bundle, config, config_json,
        empty,  empty,  reinterpret_cast<trtmc::IBackend*>(static_cast<std::uintptr_t>(1)),
        empty,  false,
    };
    try {
        (void)plugin->create(context);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

void check_ref2va_profile_abi() {
    const std::string valid_error = create_error(ref2va_profile_json());
    check(valid_error.find("missing processor/preprocessor_config.json") != std::string::npos,
          "complete Ref2VA profile reaches processor-asset validation");

    for (const auto& [name, value] : kRef2vaProfile) {
        (void)value;
        const std::string error = create_error(ref2va_profile_json(name));
        check(error.find("incompatible dynamic profile") != std::string::npos,
              "missing Ref2VA profile ABI field fails closed");
    }

    std::string legacy = ref2va_profile_json();
    const std::string current = "\"ref2va_min_condition_video_rows\":0";
    const auto position = legacy.find(current);
    check(position != std::string::npos, "Ref2VA legacy-profile fixture locates minimum");
    if (position != std::string::npos)
        legacy.replace(position, current.size(), "\"ref2va_min_condition_video_rows\":4096");
    check(create_error(legacy).find("incompatible dynamic profile") != std::string::npos,
          "legacy Ref2VA visual minimum fails closed before plan loading");
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

    check_ref2va_profile_abi();

    if (failures > 0) {
        std::cerr << failures << " MiniMax-H3 config schema test(s) failed\n";
        return 1;
    }
    return 0;
}
