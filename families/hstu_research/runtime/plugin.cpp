/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu_research/runtime/history_snapshot.h"
#include "families/hstu_research/runtime/pipeline.h"
#include "families/hstu_research/runtime/staged_pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    const auto config_bytes = context.reader.read_section("runtime.json");
    const auto config = nlohmann::json::parse(config_bytes.begin(), config_bytes.end());
    if (config.value("execution", "") == "staged") {
        const bool hydration = config.value("history_cache_schema", 0) == 1;
        const auto prepare_plan =
            context.reader.read_section(hydration ? "history_prepare.plan" : "prepare.plan");
        const auto step_plan = context.reader.read_section("step.plan");
        trtmc::ModuleCreateOptions prepare_options;
        prepare_options.engine_timing = false;
        auto prepare = context.backend.create_module(prepare_plan.data(), prepare_plan.size(),
                                                     prepare_options);
        trtmc::ModuleCreateOptions options;
        options.engine_timing = false;
        options.stream = prepare->stream();
        std::unique_ptr<trtmc::ITrtModule> request;
        if (hydration) {
            const auto plan = context.reader.read_section("request_prepare.plan");
            request = context.backend.create_module(plan.data(), plan.size(), options);
        }
        std::vector<trtmc::ModuleExternalBinding> bindings;
        for (std::size_t i = 0; i < (hydration ? 2u : 1u); ++i)
            for (const auto& spec : config.at("stages").at(i).at("outputs")) {
                const std::string name = spec.at("name");
                auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
                shape.at(0) = config.at("max_batch_size");
                bindings.push_back(
                    {name, (i ? request : prepare)->device_ptr(name),
                     trtmc::hstu_research::checked_bytes(
                         shape, trtmc::hstu_research::parse_dtype(spec.at("dtype")))});
            }
        auto step = context.backend.create_module_prebound(step_plan.data(), step_plan.size(),
                                                           options, bindings);
        if (config.value("cuda_graphs", false))
            step->enable_cuda_graph();
        const auto identity = hydration
                                  ? trtmc::hstu_research::history_file_sha256(context.reader.path())
                                  : std::string{};
        return new trtmc::hstu_research::StagedPipeline(
            std::move(prepare), std::move(step), config, context.kv_cache_size_bytes,
            context.kv_cache_memory_ratio, std::move(request), identity);
    }
    if (context.kv_cache_size_bytes != 0 || context.kv_cache_memory_ratio != 0)
        throw std::invalid_argument("HSTU monolithic methods have no runtime-managed KV cache");
    const auto plan = context.reader.read_section("engine.plan");
    if (plan.empty())
        throw std::invalid_argument("empty ranking engine");
    trtmc::ModuleCreateOptions options;
    options.engine_timing = false;
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    return new trtmc::hstu_research::Pipeline(std::move(engine), config);
}
