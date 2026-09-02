/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/foundationpose/pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "utils/json_helpers.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

std::unique_ptr<ITrtModule> load_module(const PipelineContext& context, const char* section,
                                        const char* label, cudaStream_t stream) {
    const auto* plan = find_section(context.bundle, section);
    if (plan == nullptr || plan->empty())
        throw std::runtime_error(std::string("FoundationPose bundle is missing ") + section);
    if (context.backend == nullptr)
        throw std::runtime_error("FoundationPose runtime has no TensorRT backend");
    ModuleCreateOptions options;
    options.runtime_cache_path = context.runtime_cache_path.c_str();
    options.cuda_graphs = context.cuda_graphs;
    options.stream = stream;
    auto module = context.backend->create_module(plan->data(), plan->size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("FoundationPose failed to load ") + section);
    module->set_timing_label(label);
    return module;
}

class FoundationPosePlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        auto refiner = load_module(context, "engine_plan", "FoundationPose refiner", nullptr);
        auto scorer = load_module(context, "foundationpose_score_engine_plan",
                                  "FoundationPose scorer", refiner->stream());
        if (scorer->stream() != refiner->stream())
            throw std::runtime_error("FoundationPose engines must share one CUDA stream");
        return std::make_unique<FoundationPosePipeline>(
            std::move(refiner), std::move(scorer),
            extract_json_int(context.config_json, "pose_crop_height", 160),
            extract_json_int(context.config_json, "pose_crop_width", 160),
            extract_json_int(context.config_json, "pose_crop_channels", 6),
            extract_json_int(context.config_json, "pose_refiner_max_batch", 42),
            extract_json_int(context.config_json, "pose_max_hypotheses", 252),
            context.bundle.info.model_id);
    }
};

} // namespace

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_foundationpose_plugin, FoundationPosePlugin,
                                       "foundationpose_pose_refinement");

} // namespace trtmc
