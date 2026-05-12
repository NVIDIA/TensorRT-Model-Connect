// WanPlugin: handles "diffusion_wan" strategy only.
// Uses WanPipeline with a single text encoder, denoiser, and VAE.

#include "runtime/models/wan/pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class WanPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto parts = load_diffusion_parts(ctx.backend, ctx.bundle, ctx.config_json, opts);

        // Extract first text encoder
        std::unique_ptr<TrtModule> te_module;
        if (!parts.text_encoders.empty())
            te_module = std::move(parts.text_encoders[0].module);

        return std::make_unique<WanPipeline>(std::move(te_module), std::move(parts.denoiser.module),
                                             std::move(parts.vae.module), std::move(parts.config),
                                             std::move(parts.weights), std::move(parts.tokenizer),
                                             ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_wan_plugin, WanPlugin, "diffusion_wan");

} // namespace trtmc
