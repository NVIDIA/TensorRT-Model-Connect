/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_video_diffusion/runtime/pipeline.h"
#include "families/stable_video_diffusion/runtime/runtime_config.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::stable_video_diffusion {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

ModuleLoader make_loader(IBackend& backend, const BundleReader& bundle) {
    // Plans stay on the host; each stage deserializes its engine when it runs.
    auto plans = std::make_shared<std::unordered_map<std::string, std::vector<char>>>();
    for (const char* name : {kImageEncoderPlan, kVaeEncoderPlan, kUnetPlan, kVaeDecoderPlan})
        plans->emplace(name, require_section(bundle, name));
    return [&backend, plans](const std::string& section) {
        const auto found = plans->find(section);
        if (found == plans->end())
            throw std::runtime_error("Stable Video Diffusion requested an undeclared plan: " +
                                     section);
        const ModuleCreateOptions options{};
        return backend.create_module(found->second.data(), found->second.size(), options);
    };
}

} // namespace
} // namespace trtmc::stable_video_diffusion

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    namespace svd = trtmc::stable_video_diffusion;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("stable_video_diffusion does not support --kv-cache-size");
    auto config = svd::parse_runtime_config(svd::require_section(context.reader, "runtime.json"));
    return new svd::Pipeline(svd::make_loader(context.backend, context.reader), std::move(config));
}
