/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vitpose/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::vitpose {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

} // namespace
} // namespace trtmc::vitpose

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("vitpose does not support --kv-cache-size");
    if (context.reader.info().task != trtmc::internal::IImageTextToPoints::kTask)
        throw std::runtime_error("vitpose bundle declares an unsupported task: " +
                                 context.reader.info().task);
    auto metadata = trtmc::vitpose::parse_runtime_json(
        trtmc::vitpose::require_section(context.reader, "runtime.json"));
    const auto plan = trtmc::vitpose::require_section(context.reader, "engine.plan");

    trtmc::ModuleCreateOptions options{};
    auto engine = context.backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("vitpose engine failed to load");
    return new trtmc::vitpose::VitPosePoints(std::move(engine), std::move(metadata));
}
