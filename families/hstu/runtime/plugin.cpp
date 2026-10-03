/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/cached_pipeline.h"
#include "families/hstu/runtime/native_library.h"
#include "families/hstu/runtime/pipeline.h"
#include "trtmc/history_cache.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <utility>

namespace {

trtmc::ITask* create_cached_pipeline(const trtmc::FamilyContext& context,
                                     std::unique_ptr<trtmc::ITrtModule> engine,
                                     trtmc::hstu::RuntimeConfig config) {
    trtmc::ModuleCreateOptions auxiliary_options;
    auxiliary_options.collect_timing = false;
    std::unique_ptr<trtmc::ITrtModule> prefill;
    if (context.reader.find_section("prefill.plan") != nullptr) {
        const auto prefill_plan = context.reader.read_section("prefill.plan");
        if (prefill_plan.empty())
            throw std::runtime_error("hstu prefill.plan is empty");
        prefill = context.backend.create_module(prefill_plan.data(), prefill_plan.size(),
                                                auxiliary_options);
        if (!prefill || !prefill->ok())
            throw std::runtime_error("hstu could not load prefill.plan");
        prefill->set_timing_label("hstu prefill");
    }
    std::unique_ptr<trtmc::ITrtModule> candidates;
    if (config.mode == "retrieval") {
        const auto lookup_plan = context.reader.read_section("candidate.plan");
        candidates = context.backend.create_module(lookup_plan.data(), lookup_plan.size(),
                                                   auxiliary_options);
    }
    std::shared_ptr<trtmc::HistoryCache> cache;
    if (context.kv_cache_size_bytes) {
        trtmc::HistoryCacheOptions options;
        options.max_bytes = context.kv_cache_size_bytes;
        cache = std::make_shared<trtmc::HistoryCache>(std::move(options));
    }
    return new trtmc::hstu::CachedPipeline(std::move(engine), std::move(candidates),
                                           std::move(config), std::move(cache), std::move(prefill));
}

} // namespace

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
    trtmc::ModuleCreateOptions options;
    // HSTU serving avoids per-call instrumentation; platform-created modules
    // can retain the generic default when aggregate engine timing is wanted.
    options.collect_timing = false;
    const bool native = context.reader.find_section("attention_native.json") != nullptr;
    if (native != (context.reader.find_section("attention_native.so") != nullptr))
        throw std::invalid_argument(
            "hstu native attention manifest and library must appear together");
    if (native) {
        options.plugin_libraries.push_back(trtmc::hstu::native_attention_library(
            context.reader.read_section("attention_native.json"),
            context.reader.read_section("attention_native.so"), config.enable_history_cache));
    }
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("hstu could not load engine.plan");
    engine->set_timing_label("hstu recommendation");
    if (config.enable_history_cache)
        return create_cached_pipeline(context, std::move(engine), std::move(config));
    return new trtmc::hstu::Pipeline(std::move(engine), std::move(config));
}
