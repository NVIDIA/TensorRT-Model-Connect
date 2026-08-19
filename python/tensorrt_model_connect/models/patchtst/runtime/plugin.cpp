/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// PatchTSTPlugin: handles "patchtst_trt" strategy.
// Numeric time-series models compiled from the PatchTST HF family.

#include "pipeline.h"
#include "plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace trtmc {

namespace {

std::string to_lower_copy(std::string value) {
    for (auto& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string classify_task_hint(const std::string& hint) {
    if (hint.find("class") != std::string::npos)
        return "classification";
    if (hint.find("regress") != std::string::npos)
        return "regression";
    if (hint.find("forecast") != std::string::npos || hint.find("predict") != std::string::npos)
        return "forecast";
    return "";
}

std::string infer_task_type(const std::string& json) {
    std::string task = extract_json_string(json, "patchtst_task", "");
    if (task.empty())
        task = extract_json_string(json, "task_type", "");
    if (task.empty())
        task = extract_json_string(json, "problem_type", "");

    std::string inferred = classify_task_hint(to_lower_copy(task));
    if (!inferred.empty())
        return inferred;

    const std::vector<std::string> architectures = extract_json_string_array(json, "architectures");
    for (const auto& arch : architectures) {
        inferred = classify_task_hint(to_lower_copy(arch));
        if (!inferred.empty())
            return inferred;
    }

    return "forecast";
}

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

int32_t tensor_parallel_rank_from_env() {
    for (const char* name : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK", "SLURM_PROCID"}) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0')
            return static_cast<int32_t>(std::strtol(value, nullptr, 10));
    }
    return 0;
}

} // namespace

class PatchTSTPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
        std::string engine_section = "engine_plan";
        if (tp_config.enabled) {
            engine_section = tp_engine_section_name(tensor_parallel_rank_from_env());
        }

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, engine_section), "engine_plan", opts);
        const std::string task_type = infer_task_type(ctx.config_json);
        const int32_t context_length =
            extract_json_int(ctx.config_json, "context_length", ctx.config.max_cache_length);
        const int32_t num_input_channels =
            extract_json_int(ctx.config_json, "num_input_channels", 1);
        const int32_t prediction_length = extract_json_int(ctx.config_json, "prediction_length", 0);
        const int32_t num_targets = extract_json_int(ctx.config_json, "num_targets", 1);

        return std::make_unique<PatchTSTPipeline>(
            std::move(loaded.module), task_type, context_length, num_input_channels,
            prediction_length, num_targets, ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_patchtst_plugin, PatchTSTPlugin, "patchtst_trt");

} // namespace trtmc
