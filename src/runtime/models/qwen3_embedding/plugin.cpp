/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"
#include "runtime/models/qwen3_embedding/embedding_pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <memory>

namespace trtmc {

class Qwen3EmbeddingPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;
        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        const int32_t eos_token_id =
            extract_json_int(ctx.config_json, "embedding_eos_token_id", -1);
        return std::make_unique<QwenEmbeddingPipeline>(
            std::move(loaded.module), std::move(tokenizer), eos_token_id, ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen3_embedding_plugin, Qwen3EmbeddingPlugin,
                                       "qwen_embedding");

} // namespace trtmc
