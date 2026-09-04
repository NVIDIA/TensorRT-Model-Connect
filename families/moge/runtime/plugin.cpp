/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moge/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::moge {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("moge bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load_engine(const FamilyContext& context) {
    const auto plan = require_section(context.reader, "engine.plan");
    ModuleCreateOptions options{};
    auto module = context.backend.create_module(plan.data(), plan.size(), options);
    if (!module || !module->ok())
        throw std::runtime_error("moge failed to load engine.plan");
    module->set_timing_label("moge geometry");
    return module;
}

} // namespace

ITask* create(const FamilyContext& context) {
    return new trtmc::MogePipeline(load_engine(context));
}

} // namespace trtmc::moge

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::moge::create(context);
}
