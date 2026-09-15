/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <utility>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("hstu does not support --kv-cache-size");
    const auto runtime = context.reader.read_section("runtime.json");
    std::vector<char> keys;
    if (context.reader.find_section("embedding_keys.bin") != nullptr)
        keys = context.reader.read_section("embedding_keys.bin");
    auto config = trtmc::hstu::parse_runtime_config(runtime, keys);
    const auto plan = context.reader.read_section("engine.plan");
    if (plan.empty())
        throw std::runtime_error("hstu engine.plan is empty");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("hstu could not load engine.plan");
    engine->set_timing_label("hstu recommendation");
    return new trtmc::hstu::Pipeline(std::move(engine), std::move(config));
}
