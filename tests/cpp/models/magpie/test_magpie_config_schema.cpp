/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"

#include <iostream>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

int main() {
    auto& schemas = trtmc::config::SchemaRegistry::instance();
    check(schemas.lookup("audio_magpie") == nullptr, "audio_magpie schema absent from core");

    trtmc::load_model_plugin_for_strategy("text_to_audio_magpie");
    const auto* schema = schemas.lookup("audio_magpie");
    check(schema != nullptr, "magpie plugin registers audio_magpie schema");
    check(schema != nullptr && schema->fields.size() == 6, "audio_magpie field count");

    if (failures > 0) {
        std::cerr << failures << " magpie config schema test(s) failed\n";
        return 1;
    }
    return 0;
}
