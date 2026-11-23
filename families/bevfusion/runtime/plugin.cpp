/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/bevfusion/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("bevfusion does not support --kv-cache-size");
    if (context.reader.info().task != trtmc::internal::ISpatialFeaturesToClassScores::kTask)
        throw std::invalid_argument("BEVFusion bundle has an unsupported primary Task");
    const auto* section = context.reader.find_section("engine.plan");
    if (!section || section->length == 0)
        throw std::invalid_argument("BEVFusion bundle is missing engine.plan");
    const auto plan = context.reader.read_section("engine.plan");
    auto module = context.backend.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error("BEVFusion failed to load engine.plan");
    module->set_timing_label("BEVFusion BEV segmentation");
    return new trtmc::bevfusion::Pipeline(std::move(module));
}
