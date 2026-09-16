/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/cached_pipeline.h"
#include "families/hstu/runtime/pipeline.h"
#include "trtmc/history_cache.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <utility>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    const auto runtime = context.reader.read_section("runtime.json");
    std::vector<char> keys;
    if (context.reader.find_section("embedding_keys.bin") != nullptr)
        keys = context.reader.read_section("embedding_keys.bin");
    auto config = trtmc::hstu::parse_runtime_config(runtime, keys);
    if (context.kv_cache_size_bytes && !config.enable_history_cache)
        throw std::invalid_argument(
            "hstu --kv-cache-size requires enable_history_cache in the bundle");
    const auto plan = context.reader.read_section("engine.plan");
    if (plan.empty())
        throw std::runtime_error("hstu engine.plan is empty");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("hstu could not load engine.plan");
    engine->set_timing_label("hstu recommendation");
    if (config.enable_history_cache) {
        std::unique_ptr<trtmc::ITrtModule> candidates;
        if (config.mode == "retrieval") {
            const auto lookup_plan = context.reader.read_section("candidate.plan");
            candidates = context.backend.create_module(lookup_plan.data(), lookup_plan.size(), {});
        }
        std::shared_ptr<trtmc::HistoryCache> cache;
        if (context.kv_cache_size_bytes) {
            trtmc::HistoryCacheOptions options;
            options.max_bytes = context.kv_cache_size_bytes;
            cache = std::make_shared<trtmc::HistoryCache>(std::move(options));
        }
        return new trtmc::hstu::CachedPipeline(std::move(engine), std::move(candidates),
                                               std::move(config), std::move(cache));
    }
    return new trtmc::hstu::Pipeline(std::move(engine), std::move(config));
}
