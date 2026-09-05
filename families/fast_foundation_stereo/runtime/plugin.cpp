/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/fast_foundation_stereo/runtime/stereo_pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::fast_foundation_stereo {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load_module(IBackend& backend, const std::vector<char>& plan,
                                        cudaStream_t stream, const char* label) {
    ModuleCreateOptions options{};
    options.stream = stream;
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("Fast Foundation Stereo failed to load ") + label);
    return module;
}

} // namespace
} // namespace trtmc::fast_foundation_stereo

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("fast_foundation_stereo does not support --kv-cache-size");
    const auto& feature_plan =
        trtmc::fast_foundation_stereo::require_section(context.reader, "engine.plan");
    auto feature = trtmc::fast_foundation_stereo::load_module(context.backend, feature_plan,
                                                              nullptr, "feature engine.plan");
    const auto& post_plan =
        trtmc::fast_foundation_stereo::require_section(context.reader, "post.plan");
    auto post = trtmc::fast_foundation_stereo::load_module(context.backend, post_plan,
                                                           feature->stream(), "post engine plan");
    return new trtmc::FastFoundationStereoPipeline(std::move(feature), std::move(post), "");
}
