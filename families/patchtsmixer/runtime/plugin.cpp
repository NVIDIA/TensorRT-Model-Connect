/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtsmixer/runtime/pipeline.h"
#include "families/patchtsmixer/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <utility>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    const auto& plan = trtmc::patchtsmixer::require_section(context.reader, "engine.plan");
    const auto config_text =
        trtmc::patchtsmixer::require_text_section(context.reader, "runtime.json");
    auto engine = trtmc::patchtsmixer::load_engine(context.backend, plan);
    auto config = trtmc::patchtsmixer::parse_runtime_config(config_text);
    return new trtmc::patchtsmixer::Pipeline(std::move(engine), config);
}
