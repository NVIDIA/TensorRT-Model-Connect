/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/lerobot_act/pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "utils/json_helpers.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

std::unique_ptr<ITrtModule> load_policy(const PipelineContext& context) {
    const auto* plan = find_section(context.bundle, "engine_plan");
    if (plan == nullptr || plan->empty())
        throw std::runtime_error("LeRobot ACT bundle is missing engine_plan");
    if (context.backend == nullptr)
        throw std::runtime_error("LeRobot ACT runtime has no TensorRT backend");
    ModuleCreateOptions options;
    options.runtime_cache_path = context.runtime_cache_path.c_str();
    options.cuda_graphs = context.cuda_graphs;
    auto module = context.backend->create_module(plan->data(), plan->size(), options);
    if (!module || !module->ok())
        throw std::runtime_error("LeRobot ACT failed to load engine_plan");
    module->set_timing_label("LeRobot ACT policy");
    return module;
}

} // namespace

class LeRobotActPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        const auto image_height =
            extract_json_int(ctx.config_json, "observation_image_height", 480);
        const auto image_width = extract_json_int(ctx.config_json, "observation_image_width", 640);
        const auto image_channels =
            extract_json_int(ctx.config_json, "observation_image_channels", 3);
        const auto state_dim = extract_json_int(ctx.config_json, "observation_state_dim", 14);
        const auto action_dim = extract_json_int(ctx.config_json, "action_dim", 14);
        const auto chunk_size = extract_json_int(ctx.config_json, "action_chunk_size", 100);
        auto action_min = extract_json_float_array(ctx.config_json, "action_training_min");
        auto action_max = extract_json_float_array(ctx.config_json, "action_training_max");
        if (action_min.empty() || action_max.empty())
            throw std::runtime_error("LeRobot ACT bundle is missing action training bounds");

        return std::make_unique<LeRobotActPipeline>(
            load_policy(ctx), image_height, image_width, image_channels, state_dim, action_dim,
            chunk_size, std::move(action_min), std::move(action_max), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_lerobot_act_plugin, LeRobotActPlugin,
                                       "lerobot_act_action_chunk");

} // namespace trtmc
