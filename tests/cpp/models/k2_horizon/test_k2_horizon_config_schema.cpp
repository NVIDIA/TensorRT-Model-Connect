/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/config/cli_support.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
bool rejects(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using trtmc::config::ConfigBundle;
    using trtmc::config::Layer;
    using trtmc::config::SchemaRegistry;

    auto& schemas = SchemaRegistry::instance();
    check(schemas.lookup("k2_horizon") == nullptr, "K2 schema is absent from shared core");

    trtmc::load_model_plugin_for_strategy("k2_horizon_decoder_kv_cache");
    const auto* schema = schemas.lookup("k2_horizon");
    check(schema != nullptr, "K2 DSO registers its model-owned schema");
    check(schema != nullptr && schema->fields.size() == 1, "K2 schema field count is exact");

    const ConfigBundle defaults = ConfigBundle::build({}, schemas);
    check(defaults.source_of("k2_horizon", "emit_prompt_token_ids") == Layer::SchemaDefault,
          "prompt receipt is schema-defaulted");
    check(!defaults.get<bool>("k2_horizon", "emit_prompt_token_ids"),
          "prompt receipt defaults off");

    const ConfigBundle enabled = trtmc::config::resolve_cli_config(
        "", {"k2_horizon.emit_prompt_token_ids=true"}, {}, schemas);
    check(enabled.source_of("k2_horizon", "emit_prompt_token_ids") == Layer::SessionRequest,
          "prompt receipt requires an explicit session request");
    check(enabled.get<bool>("k2_horizon", "emit_prompt_token_ids"),
          "explicit session request enables prompt receipt");

    check(rejects([&] {
              (void)trtmc::config::resolve_cli_config("", {"emit_prompt_token_ids=true"}, {},
                                                      schemas);
          }),
          "unqualified prompt receipt option is rejected");
    check(rejects([&] {
              (void)trtmc::config::resolve_cli_config(
                  "", {"k2_horizon.emit_prompt_token_ids=maybe"}, {}, schemas);
          }),
          "invalid Boolean prompt receipt option is rejected");

    if (failures != 0) {
        std::cerr << failures << " K2-Horizon config schema test(s) failed\n";
        return 1;
    }
    return 0;
}
