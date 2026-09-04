/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/foundationpose/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::foundationpose {
namespace {

std::vector<char> require_section(const BundleReader& reader, const char* name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("FoundationPose bundle is missing " + std::string(name));
    return reader.read_section(name);
}

std::unique_ptr<ITrtModule> load_module(IBackend& backend, const std::vector<char>& plan,
                                        const char* label, cudaStream_t stream) {
    ModuleCreateOptions options{};
    options.stream = stream;
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (module == nullptr || !module->ok())
        throw std::runtime_error(std::string("FoundationPose failed to load ") + label);
    module->set_timing_label(label);
    return module;
}

} // namespace

ITask* create(const FamilyContext& context) {
    const auto refiner_plan = require_section(context.reader, "engine.plan");
    const auto scorer_plan = require_section(context.reader, "score.plan");
    auto refiner = load_module(context.backend, refiner_plan, "FoundationPose refiner", nullptr);
    auto scorer =
        load_module(context.backend, scorer_plan, "FoundationPose scorer", refiner->stream());
    if (scorer->stream() != refiner->stream())
        throw std::runtime_error("FoundationPose engines must share one CUDA stream");
    return new FoundationPosePipeline(std::move(refiner), std::move(scorer), 160, 160, 6, 42, 252,
                                      10);
}

} // namespace trtmc::foundationpose

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::foundationpose::create(context);
}
