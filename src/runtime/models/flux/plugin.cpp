// FluxPlugin: handles "diffusion_flux" strategy.
// FLUX diffusion pipeline with T5 + CLIP text encoders, denoiser, and VAE.

#include "runtime/models/flux/pipeline.h"
#include "runtime/plugins/shared/diffusion_helpers.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"

namespace trtmc {

class FluxPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto parts = load_diffusion_parts(ctx.backend, ctx.bundle, ctx.config_json, opts);

        // Move text encoder modules into vector
        std::vector<std::unique_ptr<TrtModule>> te_modules;
        for (auto& te : parts.text_encoders)
            te_modules.push_back(std::move(te.module));

        // Create native BPE CLIP tokenizer from bundle
        auto clip_tok = create_clip_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<FluxPipeline>(
            std::move(te_modules), std::move(parts.denoiser.module), std::move(parts.vae.module),
            std::move(parts.config), std::move(parts.weights), std::move(parts.tokenizer),
            std::move(clip_tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_flux_plugin, FluxPlugin, "diffusion_flux");

} // namespace trtmc
