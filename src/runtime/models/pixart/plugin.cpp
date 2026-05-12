// PixArtPlugin: handles "diffusion_pixart" strategy.
// PixArt-Sigma/Alpha via TRT Network API. Same engine format as Wan
// (preprocessor_weights, T5 text encoder, DiT denoiser, VAE decoder).
// Standalone plugin — no delegation to WanPlugin.

#include "runtime/models/pixart/pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class PixArtPlugin final : public IPipelinePlugin {
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

        return std::make_unique<PixArtPipeline>(
            std::move(te_module), std::move(parts.denoiser.module), std::move(parts.vae.module),
            std::move(parts.config), std::move(parts.weights), std::move(parts.tokenizer),
            ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_pixart_plugin, PixArtPlugin, "diffusion_pixart");

} // namespace trtmc
