/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_pipeline.h"

#include "bundle/bundle_format.h"
#include "runtime/models/sam2/sam2_engine_contract.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace trtmc::sam2 {

namespace {

const std::vector<char>& requirePlan(const BundleFile& bundle, std::string_view name) {
    const auto found =
        std::find_if(bundle.sections.begin(), bundle.sections.end(),
                     [name](const BundleSection& section) { return section.name == name; });
    if (found == bundle.sections.end() || found->data.empty())
        throw std::invalid_argument("SAM2 bundle is missing nonempty plan section: " +
                                    std::string(name));
    if (std::count_if(bundle.sections.begin(), bundle.sections.end(),
                      [name](const BundleSection& section) { return section.name == name; }) != 1) {
        throw std::invalid_argument("SAM2 bundle contains a duplicate plan section: " +
                                    std::string(name));
    }
    return found->data;
}

std::array<const std::vector<char>*, 6> validateBundleInventory(const BundleFile& bundle) {
    std::size_t plan_count = 0;
    std::size_t config_count = 0;
    for (const auto& section : bundle.sections) {
        if (std::find(kRequiredPlanSections.begin(), kRequiredPlanSections.end(), section.name) !=
            kRequiredPlanSections.end()) {
            ++plan_count;
        } else if (section.name == kConfigSection) {
            ++config_count;
            if (section.data.empty())
                throw std::invalid_argument("SAM2 bundle contains an empty config.json");
        } else {
            throw std::invalid_argument("SAM2 bundle contains an unsupported section: " +
                                        section.name);
        }
    }
    if (plan_count != kRequiredPlanSections.size() || config_count != 1U ||
        bundle.sections.size() != kRequiredPlanSections.size() + 1U) {
        throw std::invalid_argument("SAM2 bundle must contain only six plans and config.json");
    }

    std::array<const std::vector<char>*, 6> plans;
    for (std::size_t index = 0; index < plans.size(); ++index)
        plans[index] = &requirePlan(bundle, kRequiredPlanSections[index]);
    return plans;
}

} // namespace

NativeVideoEngineSet makeNativeVideoEngineSet(const BundleFile& bundle,
                                              const NativePlanModuleFactory& module_factory) {
    if (!module_factory)
        throw std::invalid_argument("SAM2 requires a TensorRT module factory");
    if (bundle.info.family != "sam2" || bundle.info.runtime_strategy != kStrategyName) {
        throw std::invalid_argument("SAM2 bundle family or runtime strategy is invalid");
    }

    const auto plans = validateBundleInventory(bundle);

    std::array<std::unique_ptr<ITrtModule>, 6> modules;
    for (std::size_t index = 0; index < modules.size(); ++index) {
        const auto& plan = *plans[index];
        modules[index] = module_factory(kRequiredPlanSections[index], plan.data(), plan.size());
        if (modules[index] == nullptr || !modules[index]->ok())
            throw std::runtime_error("SAM2 failed to create TensorRT module for " +
                                     std::string(kRequiredPlanSections[index]));
    }

    NativeVideoEngineSet engines;
    engines.image = std::move(modules[0]);
    engines.prompt = std::move(modules[1]);
    for (std::size_t index = 0; index < engines.recurrent.size(); ++index)
        engines.recurrent[index] = std::move(modules[index + 2]);
    return engines;
}

std::unique_ptr<Sam2Pipeline> Sam2Pipeline::create(const PipelineContext& context,
                                                   const NativePlanModuleFactory& module_factory) {
    if (context.config.runtime_strategy != kStrategyName)
        throw std::invalid_argument("SAM2 pipeline received the wrong runtime strategy");
    auto engines = makeNativeVideoEngineSet(context.bundle, module_factory);
    auto processor = std::make_unique<NativeVideoProcessor>(std::move(engines));
    return std::unique_ptr<Sam2Pipeline>(new Sam2Pipeline(std::move(processor)));
}

Sam2Pipeline::Sam2Pipeline(std::unique_ptr<NativeVideoProcessor> processor)
    : processor_(std::move(processor)) {
    if (processor_ == nullptr)
        throw std::invalid_argument("SAM2 pipeline requires a native video processor");
}

const char* Sam2Pipeline::model_id() const {
    return kModelId.data();
}

std::unique_ptr<NativeVideoProcessor> Sam2Pipeline::releaseVideoProcessor() {
    if (processor_ == nullptr)
        throw std::logic_error("SAM2 pipeline video processor was already consumed");
    return std::move(processor_);
}

} // namespace trtmc::sam2
