/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/models/fast_foundation_stereo/stereo_pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

std::unique_ptr<ITrtModule> load_module(IBackend* backend, const std::vector<char>* plan,
                                        const ModuleCreateOptions& options, const char* label) {
    if (backend == nullptr || plan == nullptr || plan->empty())
        throw std::runtime_error(std::string("Fast Foundation Stereo missing ") + label);
    auto module = backend->create_module(plan->data(), plan->size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("Fast Foundation Stereo failed to load ") + label);
    return module;
}

} // namespace

class FastFoundationStereoPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions feature_options;
        feature_options.runtime_cache_path = ctx.runtime_cache_path.c_str();
        feature_options.cuda_graphs = ctx.cuda_graphs;
        auto feature = load_module(ctx.backend, find_section(ctx.bundle, "engine_plan"),
                                   feature_options, "feature engine_plan");

        ModuleCreateOptions post_options = feature_options;
        post_options.stream = feature->stream();
        auto post = load_module(ctx.backend,
                                find_section(ctx.bundle, "fast_foundation_stereo_post_engine_plan"),
                                post_options, "post engine plan");
        return std::make_unique<FastFoundationStereoPipeline>(std::move(feature), std::move(post),
                                                              ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_fast_foundation_stereo_plugin,
                                       FastFoundationStereoPlugin,
                                       "fast_foundation_stereo_disparity");

} // namespace trtmc
