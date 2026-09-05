/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtsmixer/runtime/pipeline.h"
#include "families/patchtsmixer/runtime/plugin_helpers.h"
#include "trtmc/runtime/family_factory.h"

#include <stdexcept>
#include <string>
#include <utility>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("patchtsmixer does not support --kv-cache-size");
    const auto config_text =
        trtmc::patchtsmixer::require_text_section(context.reader, "runtime.json");
    auto config = trtmc::patchtsmixer::parse_runtime_config(config_text);
    const auto rank = trtmc::patchtsmixer::require_rank(config.tensor_parallel_size);
    const std::string section = config.tensor_parallel_size == 1
                                    ? "engine.plan"
                                    : "engine.rank" + std::to_string(rank) + ".plan";
    const auto& plan = trtmc::patchtsmixer::require_section(context.reader, section.c_str());
    auto engine = trtmc::patchtsmixer::load_engine(context.backend, plan);
    return new trtmc::patchtsmixer::Pipeline(std::move(engine), config);
}
