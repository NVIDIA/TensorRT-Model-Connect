// EncoderPlugin: handles "encoder_only", "embedding", "reranking", and
// "neural_operator" strategies. Single-pass encoder models (BERT, Eagle, etc.).

#include "runtime/models/encoder/pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class EncoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "engine_plan", opts);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<EncoderPipeline>(std::move(loaded.module),
                                                 ctx.config.runtime_strategy, std::move(tokenizer),
                                                 ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_encoder_plugin, EncoderPlugin, "encoder_only",
                                       "embedding", "reranking", "neural_operator");

} // namespace trtmc
