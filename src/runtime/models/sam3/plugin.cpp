/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Sam3Plugin: SAM3-owned text-prompted segmentation strategy.

#include "plugin_helpers.h"
#include "runtime/models/sam3/sam3_pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {

namespace {

Sam3Config make_sam3_config(const std::string& json) {
    Sam3Config cfg;
    cfg.text_max_position_embeddings = extract_json_int(json, "sam3_text_max_position_embeddings",
                                                        cfg.text_max_position_embeddings);
    cfg.text_pad_token_id = extract_json_int(json, "sam3_text_pad_token_id", cfg.text_pad_token_id);
    cfg.text_projection_dim =
        extract_json_int(json, "sam3_text_projection_dim", cfg.text_projection_dim);
    cfg.image_size = extract_json_int(json, "sam3_image_size",
                                      extract_json_int(json, "input_image_h", cfg.image_size));
    cfg.low_res_mask_size = extract_json_int(json, "sam3_low_res_mask_size", cfg.low_res_mask_size);
    cfg.num_queries = extract_json_int(json, "sam3_num_queries", cfg.num_queries);
    cfg.score_threshold = extract_json_float(json, "sam3_score_threshold", cfg.score_threshold);
    cfg.mask_threshold = extract_json_float(json, "sam3_mask_threshold", cfg.mask_threshold);

    auto mean = extract_json_float_array(json, "image_mean", 3);
    if (mean.size() == 3)
        cfg.image_mean = std::move(mean);
    auto stdv = extract_json_float_array(json, "image_std", 3);
    if (stdv.size() == 3)
        cfg.image_std = std::move(stdv);
    return cfg;
}

} // namespace

class Sam3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto text_encoder = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "sam3 text_encoder", opts);
        auto vision_encoder =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, "vision_engine_plan"),
                                      "sam3 vision_encoder", opts);
        auto core_engine = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "sam3_core_engine_plan"), "sam3 core_engine",
            opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        return std::make_unique<Sam3Pipeline>(
            std::move(text_encoder.module), std::move(vision_encoder.module),
            std::move(core_engine.module), std::move(tokenizer), make_sam3_config(ctx.config_json),
            ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sam3_plugin, Sam3Plugin,
                                       "sam3_prompted_segmentation");

} // namespace trtmc
