/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/moge/pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <stdexcept>

namespace trtmc {
namespace {

std::unique_ptr<ITrtModule> load_moge_engine(const PipelineContext& context) {
    const auto* plan = find_section(context.bundle, "engine_plan");
    if (plan == nullptr || plan->empty())
        throw std::runtime_error("MoGe bundle is missing engine_plan");
    if (context.backend == nullptr)
        throw std::runtime_error("MoGe runtime has no TensorRT backend");

    ModuleCreateOptions options;
    options.runtime_cache_path = context.runtime_cache_path.c_str();
    options.cuda_graphs = context.cuda_graphs;
    auto module = context.backend->create_module(plan->data(), plan->size(), options);
    if (!module || !module->ok())
        throw std::runtime_error("MoGe failed to load engine_plan");
    module->set_timing_label("engine_plan");
    return module;
}

} // namespace

class MogePlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        return std::make_unique<MogePipeline>(load_moge_engine(context),
                                              context.bundle.info.model_id,
                                              context.bundle.info.precision == "fp16");
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_moge_plugin, MogePlugin, "moge_monocular_geometry");

} // namespace trtmc
