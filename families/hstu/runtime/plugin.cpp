/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/history_snapshot.h"
#include "families/hstu/runtime/staged_pipeline.h"
#include "families/hstu/runtime/validation.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
#if defined(HSTU_TRITON_026_ABI)
    const auto cache_bytes = context.kv_cache_size_bytes;
    const auto cache_ratio = context.kv_cache_memory_ratio;
#else
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("hstu does not support --kv-cache-size");
    const std::uint64_t cache_bytes = 0;
    const double cache_ratio = 0;
#endif
    const auto config_bytes = context.reader.read_section("runtime.json");
    const auto config = nlohmann::json::parse(config_bytes.begin(), config_bytes.end());
    if (config.value("execution", "") != "staged" || config.value("history_cache_schema", 0) != 1)
        throw std::invalid_argument("unsupported DLRMv3 cache graph contract");
    const auto prepare_plan = context.reader.read_section("history_prepare.plan");
    const auto request_plan = context.reader.read_section("request_prepare.plan");
    const auto step_plan = context.reader.read_section("step.plan");
    trtmc::ModuleCreateOptions options;
    auto prepare = context.backend.create_module(prepare_plan.data(), prepare_plan.size(), options);
    options.stream = prepare->stream();
    auto request = context.backend.create_module(request_plan.data(), request_plan.size(), options);
    std::vector<trtmc::ModuleExternalBinding> bindings;
    for (std::size_t i = 0; i < 2; ++i)
        for (const auto& spec : config.at("stages").at(i).at("outputs")) {
            const std::string name = spec.at("name");
            auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
            shape.at(0) = config.at("max_batch_size");
            bindings.push_back(
                {name, (i ? request : prepare)->device_ptr(name),
                 trtmc::hstu::checked_bytes(shape, trtmc::hstu::parse_dtype(spec.at("dtype")))});
        }
    auto step = context.backend.create_module_prebound(step_plan.data(), step_plan.size(), options,
                                                       bindings);
    if (config.value("cuda_graphs", false))
        step->enable_cuda_graph();
    return new trtmc::hstu::StagedPipeline(std::move(prepare), std::move(step), config, cache_bytes,
                                           cache_ratio, std::move(request),
                                           config.at("bundle_id").get<std::string>());
}
